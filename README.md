# CIFAR-10 ResNet Lab

> A reproducible image-classification pipeline with a compact residual CNN, deterministic train/validation split, and held-out test evaluation.

## Goal

Compact residual CNN trained on CIFAR-10 with a held-out validation split and one final test evaluation. No pretrained weights or fabricated performance claims.

## Dataset

CIFAR-10, 60,000 32×32 color images in 10 classes. torchvision downloads the official archive on first run and verifies its published checksum. Dataset files are kept under `data/` and excluded from Git.

**Official source:** [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
**Data terms:** Review the dataset terms on the source page before redistributing the image files. This repository downloads the dataset at run time and does not redistribute it.

Raw dataset files are not committed. First run downloads data into the ignored `data/` directory.

## Run

Python 3.10 or newer is required. Create a virtual environment, install the local package, and train:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
python -m cifar10_lab.train --epochs 20 --data-dir data --output-dir artifacts
```

The first run may download the dataset. Checkpoint and metrics output is written to `artifacts/`.

## Reproducibility and evaluation

- Training, validation, and test responsibilities are documented in the training code.
- Random seeds and key hyperparameters are explicit CLI options or constants.
- Test metrics are reported only after model selection on validation data.
- Results are not published in this README until a reproducible run has produced them.

## Stack

Computer vision · PyTorch · residual networks · reproducible training

## Scope

This is an independently built portfolio project using public benchmark data. It contains no client records, employer code, or claims that it reproduces private production systems. Dataset attribution and license obligations remain with users of the data.

## Validation status

Python source syntax has been checked. Dependencies have not been installed and training has not been run; runtime behavior and model quality are not yet verified.
