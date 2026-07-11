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

For exact reproduction of the manuscript results, install the frozen environment:

```bash
pip install -r requirements-frozen.txt
pip install nicheverse==0.1.0
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

Training the 173 sample cohort (5.66M cells, 366 genes) takes roughly 25 minutes per epoch on a single NVIDIA A100 with batch size 2048; 300 epochs run in roughly five hours. Inference (predict) on the same cohort takes under 10 minutes on the same GPU and roughly 90 minutes on a 32 core CPU. RAM use peaks around 40 GB during k-NN graph construction; you can reduce this by pre splitting the cohort by sample and concatenating outputs.
