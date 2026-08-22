# Antarctic Ice-Shelf Basal Melt: Tipping Points and Graph Neural Network Emulators

Code for a study of abrupt, non-linear transitions in the basal melt of Antarctic
ice shelves. The work combines two complementary models:

1. **A low-dimensional cavity box model** (`aisgnn.boxmodel`) resolving the coupled
   cavity / coastal-polynya system. It reproduces the cold (convective, DSW-ventilated)
   and warm (diffusive, mWDW-ventilated) circulation regimes, and the positive feedback
   between them that produces bistability. Numerical continuation traces the full
   S-shaped bifurcation diagram, including the unstable branch, and locates saddle-node
   bifurcations, hysteresis widths and rate-induced tipping thresholds.

2. **Graph neural network emulators** (`aisgnn.models`) trained on cavity-resolving
   ocean model output. Their learned spatial connectivity is used to diagnose the length
   scales over which upstream ocean variability controls downstream melt, and how those
   scales change as a cavity approaches a regime shift.

The box model provides independent ground truth against which the emulator's
reconstructed bifurcation structure is checked.

## Scientific questions

| | Question |
|---|---|
| H1 | Over what length scales does upstream ocean variability control basal melt, and how does that change with warming? |
| H2 | How does emulator skill degrade across ocean models with different parameterisations? |
| H3 | Do the dominant melt controls shift from thermal driving to circulation under warming? |
| H4 | Do changes in spatial connectivity precede a melt-regime shift? |
| H5 | What is the bifurcation structure of ice-shelf cavities — thresholds, hysteresis width, rate-induced tipping? |

## Layout

```
aisgnn/
  config.py        paths, dataset registry, physical constants
  boxmodel/        cavity model, continuation, rate-induced tipping, calibration
  data/            Zenodo download, NEMO and MISOMIP2 loaders, features, graphs
  models/          MLP baseline, GCN, GAT, edge-conditioned GNN, deep ensembles
  dynsys/          emulator sweeps, bifurcation detection, early-warning signals
  interpret/       attention-based connectivity, SHAP, intervention analysis
  coupling/        reduced flowline ice model for grounding-line response
  viz/             figure styling and panels
scripts/           numbered pipeline stages, runnable from the command line
slurm/             job scripts for the HPC
tests/             unit tests
```

## Data

All input data are public. `scripts/00_download_data.py` fetches them from Zenodo with
checksum verification and resumable transfers.

| Dataset | Source |
|---|---|
| Circum-Antarctic NEMO, present-day + REPEAT1970 + 4×CO₂ | Burgard et al. (2023), Zenodo `10149919` |
| NEMO 5 km fields and cavity geometry | Burgard et al. (2022), Zenodo `7308352` |
| MISOMIP2 OceanA-hind, ROMS-UTAS | Zenodo `21728621` |
| MISOMIP2 Ocean-hind, IGE NEMO4.0 | Zenodo `21514655` |
| MISOMIP2 Ocean-hind, UCLA-UMD MITgcm | Zenodo `21626519` |
| MISOMIP2 MIPkit-A observations | Zenodo `21679622` |

The three MISOMIP2 models ran the same protocol on the same Amundsen domain, so
cross-model comparison isolates parameterisation differences with forcing and geometry
held fixed.

## Installation

```bash
conda env create -f environment.yml
conda activate aisgnn
pip install -e .
```

Set `AISGNN_ROOT` to the directory holding `data/`, `runs/` and `figures/`:

```bash
export AISGNN_ROOT=/path/to/project
```

## Usage

```bash
python scripts/00_download_data.py --list          # show registered datasets
python scripts/00_download_data.py                 # fetch everything (~36 GB)
python scripts/09_boxmodel_reference.py            # calibrate and run the box model
python scripts/01_preprocess_nemo.py               # build per-shelf fields
python scripts/02_build_graphs.py                  # construct graphs
python scripts/03_train.py --arch gat --seed 0     # train an emulator
python scripts/08_h5_bifurcation.py                # emulator bifurcation diagrams
```

On the HPC, the equivalent SLURM scripts are in `slurm/`.

## Requirements

Python 3.11, PyTorch with CUDA, PyTorch Geometric, xarray, SciPy, SHAP, Captum.
A GPU with ≥16 GB is needed for emulator training; the box model runs on a laptop.

## Licence

MIT, see `LICENSE`.
