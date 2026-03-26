# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ReLERNN (Recombination Landscape Estimation using Recurrent Neural Networks) uses bidirectional GRU networks trained on coalescent simulations (msprime) to predict per-base recombination rates from VCF or pool-seq data.

## Setup & Installation

```bash
pixi install          # GPU (default)
pixi install -e cpu   # CPU-only
```

All dependencies (Python, TensorFlow, CUDA, Keras, etc.) are managed by pixi via `pixi.toml`. Package metadata and CLI entry points are defined in `pyproject.toml`.

## Running the Pipeline

The tool has two parallel pipelines (VCF for individual genotypes, POOL for pool-seq data), each with 4 stages run sequentially:

```bash
# VCF pipeline
ReLERNN_SIMULATE  -v input.vcf -g genome.bed -d ./output/ --assumedMu 1e-8
ReLERNN_TRAIN     -d ./output/
ReLERNN_PREDICT   -v input.vcf -d ./output/
ReLERNN_BSCORRECT -d ./output/                # optional: bootstrap CIs

# Pool-seq pipeline (same pattern with _POOL suffix, uses --pool instead of --vcf)
ReLERNN_SIMULATE_POOL --pool input.pool --sampleDepth 20 -g genome.bed -d ./output/
ReLERNN_TRAIN_POOL    -d ./output/ --readDepth 20 --maf 0.05
ReLERNN_PREDICT_POOL  --pool input.pool -d ./output/
ReLERNN_BSCORRECT     -d ./output/
```

Example pipelines with test data: `pixi run example` and `pixi run example-pool`.

There is no test suite. Validate changes by running the example pipelines end-to-end.

## Architecture

**CLI entry points** (`ReLERNN/ReLERNN_*`) — 7 executable scripts registered via pyproject.toml. These are not `.py` files but are importable Python scripts that compose the core modules.

**Core modules** (all in `ReLERNN/`):
- **manager.py** — `Manager` class: VCF/pool file splitting (by chromosome), HDF5 conversion, window size calculation. Heavy use of multiprocessing.
- **simulator.py** — `Simulator` class: runs msprime coalescent simulations to generate training data (.npy genotype matrices + pickle metadata). Supports demographic histories (stairwayplot/SMC++/MSMC), recombination hotspots, phasing errors, and accessibility masks.
- **networks.py** — Keras model definitions: `GRU_TUNED84` (VCF: bidirectional GRU + position embedding), `GRU_POOLED` (pool-seq), `HOTSPOT_CLASSIFY` (hotspot detection).
- **sequenceBatchGenerator.py** — `SequenceBatchGenerator` (extends `keras.utils.Sequence`): data batching, padding, allele frequency conversion, sample sorting by genetic similarity, z-score normalization.
- **helpers.py** — Multiprocessing utilities, window size binary search, mask statistics, demographic history parsing.
- **imports.py** — Centralized imports used by all modules.

**Data flow:** SIMULATE reads input → splits by chromosome → runs msprime to create train/vali/test .npy files → TRAIN loads via SequenceBatchGenerator → trains GRU → saves model.keras → PREDICT loads model and real data → outputs per-window recombination rates → BSCORRECT adds confidence intervals via parametric bootstrap.

**Key serialization formats:** pickle (.p) for simulation metadata, NumPy (.npy) for genotype/position matrices, HDF5 for VCF storage, Keras 3 native format (.keras) for trained models.
