# Installation

## Requirements

Python 3.10 or later, PyTorch 2.0 or later, and a working scanpy installation. A CUDA capable GPU is optional but speeds up training by 10 to 50x.

## Install from PyPI

```bash
pip install nicheverse
```

## Install from source

```bash
git clone https://github.com/digvijayky/nicheverse
cd nicheverse
pip install -e .
```

To add the developer tools (pytest, ruff):

```bash
pip install -e ".[dev]"
```

## Pinned environment for reproducibility

For exact reproduction of the manuscript results, install the frozen environment.
The manuscript reference results come from the 0.2.0 default configuration (the
b32k checkpoint), which is the current release; pin to the release matching the
reproducibility bundle:

```bash
pip install -r requirements-frozen.txt
pip install nicheverse==0.2.0
```

`requirements-frozen.txt` ships in the source distribution and the Zenodo bundle.

## GPU setup

CUDA is auto detected at runtime. If you want to force CPU on a GPU machine, pass `--device cpu` to the CLI or `device="cpu"` to the Python API. The default seed is 9 and determinism is on; on GPU we set `torch.backends.cudnn.deterministic=True` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` to enable bit identical runs across invocations on the same hardware.

## Verifying the install

```bash
python -c "import nicheverse; print(nicheverse.__version__)"
nicheverse --help
pytest -q   # if installed from source with the dev extras
```

The full test suite should pass in well under a minute on CPU.

## Hardware notes

Training the 173 sample cohort (5.66M cells, 366 genes) at the b32k reference configuration (batch size 32768, seed 9, spatial graph `knn_radius` at radius 50 microns, k_neighbors 20) takes about 17 seconds per epoch on a single NVIDIA A100, so 300 epochs complete in about 1.4 hours at a peak GPU memory of about 43 GB. Inference (predict) on the same cohort takes under 10 minutes on that GPU and scales with cohort size on a CPU (the neighborhood graph build dominates). RAM use peaks around 40 GB during k-NN graph construction; you can reduce this by pre splitting the cohort by sample and concatenating outputs.
