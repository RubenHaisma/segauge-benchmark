"""Run the benchmark on Modal GPUs.

Each model runs in its OWN Modal image (with that model's heavy deps) so their
environments stay isolated, exactly as on a local machine. The harness source
(segbench) and segauge are mounted from the sibling checkouts, so no PyPI publish
is needed. Datasets and model weights live in Modal Volumes so they persist.

Flow: a cheap CPU step downloads the dataset into the volume once (so the GPU
workers don't race to download), then one GPU function per model runs inference +
segauge scoring and returns its partial result. The results are merged and the
ranking analysis + site are produced locally.

    uv run modal run modal_app.py --config configs/kits23_modal.yaml --n-cases 5
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent
SEGAUGE_PKG = REPO.parent / "segauge" / "src" / "segauge"
SEGBENCH_PKG = REPO / "src" / "segbench"

app = modal.App("segauge-benchmark")

data_vol = modal.Volume.from_name("segbench-data", create_if_missing=True)
weights_vol = modal.Volume.from_name("segbench-weights", create_if_missing=True)
VOLS = {"/data": data_vol, "/weights": weights_vol}

_BASE_PIP = [
    "numpy", "scipy", "scikit-image", "nibabel", "trimesh", "rtree",
    "jinja2", "pyyaml",
]
GPU = "A10G"  # 24 GB, comfortably fits all the models here


def _mount(image: modal.Image) -> modal.Image:
    return (
        image.env({"PYTHONPATH": "/root/pkgs"})
        .add_local_dir(str(SEGAUGE_PKG), "/root/pkgs/segauge", copy=True)
        .add_local_dir(str(SEGBENCH_PKG), "/root/pkgs/segbench", copy=True)
    )


def _slim() -> modal.Image:
    return modal.Image.debian_slim(python_version="3.11")


cpu_image = _mount(_slim().pip_install(*_BASE_PIP))

ts_image = _mount(
    _slim()
    .pip_install(*_BASE_PIP, "TotalSegmentator")
    .env({
        "SEGBENCH_TS_BIN": "TotalSegmentator",
        "TOTALSEG_HOME_DIR": "/weights/totalseg",
    })
)

ctfm_image = _mount(
    _slim()
    .pip_install(*_BASE_PIP, "torch", "monai", "lighter-zoo", "huggingface-hub")
    .env({"SEGBENCH_CTFM_PY": "python", "HF_HOME": "/weights/hf"})
)

monai_image = _mount(
    _slim()
    .pip_install(
        *_BASE_PIP, "monai[fire]==1.4.0", "torch", "itk", "pytorch-ignite", "requests"
    )
    .env({
        "SEGBENCH_MONAI_PY": "python",
        "SEGBENCH_MONAI_BUNDLE_DIR": "/weights/monai",
    })
)

moose_image = _mount(
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*_BASE_PIP, "torch", "moosez")
    .env({"SEGBENCH_MOOSE_PY": "python"})
)


def _cfg_for_container(config_dict: dict, n_cases: int):
    from segbench.run import RunConfig

    cfg = RunConfig.from_dict(config_dict)
    cfg.n_cases = n_cases
    cfg.dataset = {**cfg.dataset, "root": f"/data/{cfg.dataset['name']}"}
    cfg.cache_dir = "/data/predictions"
    return cfg


@app.function(image=cpu_image, volumes=VOLS, timeout=3600)
def prepare_data(config_dict: dict, n_cases: int) -> int:
    """Download the dataset into the volume once, before the GPU workers run."""
    from segbench.run import _prepare

    cfg = _cfg_for_container(config_dict, n_cases)
    _, _, _, cases, _ = _prepare(cfg)
    data_vol.commit()
    return len(cases)


@app.function(image=cpu_image, volumes=VOLS, timeout=14400)
def download_amos() -> dict:
    """Fetch + unpack the AMOS22 archive into the volume once (~24 GB)."""
    import json
    import os
    import urllib.request
    import zipfile

    root = "/data/amos_ct"
    os.makedirs(root, exist_ok=True)
    marker = os.path.join(root, "amos22", "dataset.json")
    if not os.path.exists(marker):
        zp = os.path.join(root, "amos22.zip")
        if not os.path.exists(zp):
            url = "https://zenodo.org/api/records/7155725/files/amos22.zip/content"
            print("[amos] downloading 24 GB archive ...")
            urllib.request.urlretrieve(url, zp)
        print("[amos] extracting ...")
        with zipfile.ZipFile(zp) as z:
            z.extractall(root)
        os.remove(zp)
        data_vol.commit()
    with open(marker) as fh:
        labels = json.load(fh).get("labels", {})
    print(f"[amos] labels: {labels}")
    return labels


@app.local_entrypoint()
def prep_amos():
    """One-time AMOS download + label verification: modal run modal_app.py::prep_amos"""
    labels = download_amos.remote()
    print("AMOS dataset.json labels:")
    for k, v in labels.items():
        print(f"  {k}: {v}")


def _run_one(config_dict: dict, model_spec: dict, n_cases: int):
    from segbench.run import run_single_model

    cfg = _cfg_for_container(config_dict, n_cases)
    result = run_single_model(cfg, model_spec)
    # Persist downloaded model weights so the next run skips the multi-GB
    # re-download. Each model writes to its own /weights/* subdir, so commits
    # rarely collide; tolerate a conflict rather than failing the run over it.
    try:
        weights_vol.commit()
    except Exception as exc:  # noqa: BLE001
        print(f"[modal] weights commit skipped: {exc}")
    return result


@app.function(image=ts_image, gpu=GPU, memory=32768, volumes=VOLS, timeout=7200)
def run_totalsegmentator(config_dict: dict, model_spec: dict, n_cases: int):
    return _run_one(config_dict, model_spec, n_cases)


@app.function(image=ctfm_image, gpu=GPU, memory=32768, volumes=VOLS, timeout=7200)
def run_ctfm(config_dict: dict, model_spec: dict, n_cases: int):
    return _run_one(config_dict, model_spec, n_cases)


# MONAI's bundle postprocessing (softmax + invert over 105 classes on the native
# grid) spikes memory on large cases, so it gets a bigger GPU.
@app.function(
    image=monai_image, gpu="A100-40GB", memory=32768, volumes=VOLS, timeout=7200
)
def run_monai(config_dict: dict, model_spec: dict, n_cases: int):
    return _run_one(config_dict, model_spec, n_cases)


@app.function(image=moose_image, gpu=GPU, memory=32768, volumes=VOLS, timeout=7200)
def run_moose(config_dict: dict, model_spec: dict, n_cases: int):
    return _run_one(config_dict, model_spec, n_cases)


MODEL_FUNCS = {
    "totalsegmentator": run_totalsegmentator,
    "ctfm": run_ctfm,
    "monai": run_monai,
    "moose": run_moose,
}


@app.local_entrypoint()
def main(config: str = "configs/kits23_modal.yaml", n_cases: int = 5):
    import yaml

    from segbench.datasets import get_dataset
    from segbench.run import RunConfig, assemble_results
    from segbench.site import render_site

    cfg_dict = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    cfg = RunConfig.from_dict(cfg_dict)
    cfg.n_cases = n_cases

    # 1. Download data once.
    n = prepare_data.remote(cfg_dict, n_cases)
    print(f"[modal] {n} case(s) prepared")

    # 2. Run every model in parallel, each in its own image.
    handles = []
    for spec in cfg_dict["models"]:
        fn = MODEL_FUNCS.get(spec["type"])
        if fn is None:
            print(f"[modal] no Modal function for model type {spec['type']!r}; skip")
            continue
        handles.append((spec, fn.spawn(cfg_dict, spec, n_cases)))

    entries, scores = [], {}
    for spec, h in handles:
        label = spec.get("name", spec["type"])
        try:
            entry, msc = h.get()
        except Exception as exc:  # noqa: BLE001 - one model crashing must not
            # lose every other model's results.
            print(f"[modal] {label} crashed remotely ({exc}); skipping")
            continue
        if entry is None:
            continue
        entries.append(entry)
        if msc:
            scores[entry["name"]] = msc

    # 3. Merge + analyze + render locally.
    ds_spec = dict(cfg.dataset)
    ds_spec.pop("name")
    ds_spec.pop("root")
    dataset = get_dataset(cfg.dataset["name"], Path(cfg.dataset["root"]), **ds_spec)
    organ_names = [
        o.name for o in dataset.schema.organs
        if not cfg.organs or o.name in cfg.organs
    ]
    results = assemble_results(dataset, n, entries, scores, organ_names, cfg)

    out = Path(cfg_dict.get("results_path", "results/results.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    render_site(results, Path("docs"))

    print(f"models: {[m['name'] for m in results['models']]}")
    for m in results["models"]:
        for organ, cell in m.get("per_organ", {}).items():
            d = cell["summary"]["dice"]
            flag = " [SUSPECT]" if cell.get("suspect") else ""
            print(f"  {m['name']} / {organ}: dice {d['value']:.3f}{flag}")
