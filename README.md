# Geometry-Aware Zone-Level Thermal Prediction

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python)](environment.yml)
[![MIT License](https://img.shields.io/badge/Code-MIT-green.svg)](LICENSE)
[![Zenodo DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23547641.svg)](https://doi.org/10.5281/zenodo.23547641)

A Python framework for generating building thermal simulation datasets and predicting hourly zone-level operative temperatures from three-dimensional building information.

The framework combines parametric building generation, geometric and physical data processing, a geometry-aware neural network, and temporal inference. Its prediction model uses sparse 3D convolution, target-zone attention over the building representation, and an autoregressive decoder that estimates the 0.1, 0.5, and 0.9 temperature quantiles.

The repository includes **20 generated building cases** with HBJSON models, processed data, and example prediction results. Its four stages can be run independently using the outputs of the preceding stage or existing compatible files.

## Context and aim

This research originated as a **Master's thesis** at Chalmers University of Technology, within the **Sustainable Built Environments** research group in the [Department of Architecture and Civil Engineering](https://www.chalmers.se/en/departments/ace/). It addresses the need for rapid, zone-level thermal assessment across buildings with different geometries and constructions, without relying on measured indoor-temperature histories from the target building.

The framework combines explicit three-dimensional building information with weather and operating conditions to predict hourly zone operative temperatures. The repository provides reusable research software, a trained model, and example data for building generation, simulation, training, and inference. The included examples are based on controlled, free-running EnergyPlus simulations rather than measured building data.

**Project status:** Initial research software release (v1.0.0, October 2026); journal manuscript in preparation.

## People and roles

| Name | Role | Affiliation | ORCID or profile |
| --- | --- | --- | --- |
| Chengrui Yang | Lead researcher and developer | Chalmers University of Technology | - |
| Alexander Hollberg | Master's thesis supervisor | Chalmers University of Technology | - |
| Sanjay Somanath | Research mentor and collaborator | Chalmers University of Technology | - |
| Yinan Yu | Master's thesis co-supervisor; research collaborator | Chalmers University of Technology | - |

## Research outputs

- **Master's thesis:** *From Spatial to Temporal: A Geometry-Aware Approach to Indoor Thermal Prediction* (Chalmers University of Technology, 2026). Link to the Chalmers thesis portal record to be added when available.
- **Archived research software and example data:** [Zenodo, version 1.0.0](https://doi.org/10.5281/zenodo.23547641). Includes source code, model weights, and 20 generated building cases with simulation and prediction outputs.
- **Journal article:** *From Spatial to Temporal: Geometry-Aware Cross-Building Prediction of Zone-Level Thermal Dynamics* (manuscript in preparation; DOI pending).
- **How to cite the software:** Cite the [version-specific Zenodo record](https://doi.org/10.5281/zenodo.23547641); its citation formats are provided on the record page. The software, thesis, and journal article are separate research outputs and may have different identifiers.

## Workflow

```text
Seed + generation parameters
         |
         v
  Building generation --------------------> HBJSON + metadata + optional plots
         |
         v
    Data processing
      |         |
      v         v
 Voxel NPZ   EnergyPlus + EPW
      |         |
      |         v
      |      Label NPZ
      |         |
      +----+----+
           |
           v
     Model training (+ weather)
           |
           v
    Checkpoint + normalization
           |
           v
     Model inference (+ building inputs and weather)
           |
           v
  Hourly zone temperatures (XLSX)
```

Each stage has its own command-line interface and can be executed separately; a single end-to-end command is not required.

## Repository structure

```text
data_generation/          Parametric building and HBJSON generation
data_pipeline/            HBJSON-to-voxel and EnergyPlus label generation
model_pipeline/           Model architecture, data loading, and training
inference/                Single-window temperature prediction
data_output/
  manifest.json           Building IDs and dataset splits
  hbjson/                 Generated buildings and metadata
  hbjson_plot/            Optional building drawings
  train_set/
    voxel/  label/
  val_set/
    voxel/  label/
  test_set/
    voxel/  label/
  results/                 Per-building prediction spreadsheets
checkpoints/main/          Selected model checkpoint and associated files
epw/                       Weather download instructions (EPW not bundled)
environment.yml            Python environment
```

## Installation

The tested software environment uses:

- Python 3.11
- PyTorch 2.5.1 with CUDA 12.1
- `spconv-cu121` 2.3.8
- EnergyPlus 25.2.0 (installed separately)

Other Python dependencies are specified in [`environment.yml`](environment.yml).

```bash
conda env create -f environment.yml
conda activate geometry-aware-thermal-prediction
```

Install EnergyPlus separately and provide the path to its executable when generating labels. The model's sparse-convolution implementation requires a compatible CUDA/spconv setup.

**GPU access:** Training and inference require a compatible NVIDIA CUDA GPU. Cloud GPU environments, such as Google Colab, can be used when a local GPU is unavailable. Inference has been successfully tested on a Colab A100 GPU.

Commands below use paths relative to the repository root.

## Weather data

For reproducing the example simulation labels and predictions, we recommend `SWE_VG_Goteborg.City.AP.025120_TMYx.2011-2025.epw`, the Gothenburg City Airport TMYx weather file based on 2011-2025 source data. Download the matching version from the [official Climate.OneBuilding Sweden directory](https://climate.onebuilding.org/WMO_Region_6_Europe/SWE_Sweden/index.html), extract the EPW, and place it at `epw/SWE_VG_Goteborg.City.AP.025120_TMYx.2011-2025.epw` before running label generation, training, or inference. See [epw/README.md](epw/README.md) for the download instructions.

The weather file is downloaded separately and is not distributed with this repository. Keep simulation labels and model predictions paired with the same EPW; `--epw` can select another file.

## Usage

### 1. Generate buildings

Generate the 20 included building cases from seeds 101-120:

```bash
python -m data_generation --start-seed 101 --count 20 \
  --output-dir data_output/hbjson \
  --plot-dir data_output/hbjson_plot \
  --config data_generation/configs/free_running.yaml
```

The generator creates HBJSON buildings and accompanying metadata. The supplied `free_running.yaml` preset uses zero internal equipment, occupant, and HVAC gains. Omit `--plot-dir` if drawings are not needed.

Generation settings are defined in `data_generation/configs/`, `data_generation/geometric_generation/geometric_config.py`, and `data_generation/material_generation/material_config.py`. The default parameter ranges are informed by the typical configurations summarized in [data_generation/REFERENCES.md](data_generation/REFERENCES.md).

### 2. Generate voxel representations and simulation labels

```bash
python -m data_pipeline --mode all \
  --hbjson-dir data_output/hbjson \
  --manifest data_output/manifest.json \
  --split-root data_output \
  --epw epw/SWE_VG_Goteborg.City.AP.025120_TMYx.2011-2025.epw \
  --energyplus-exe /path/to/energyplus
```

The two branches can also be run separately with `--mode voxel` or `--mode label`. Without a manifest, specify the output destinations using `--voxel-out-dir` and `--label-out-dir`.

- **Voxel files (`.npz`)** store spatial and physical attributes together with zone-to-voxel mappings. The model uses 12 input features: eight continuous physical values and four boundary-category indicators.
- **Label files (`.npz`)** store hourly simulation outputs, including operative temperatures, and the building/zone information needed by the model. A companion `.xlsx` file provides outdoor temperature and zone-level operative temperatures for convenient inspection.

Voxel and label files are matched by building ID. EnergyPlus labels are generated from HBJSON-derived IDF models and weather data, independently of the voxelization branch.

### 3. Train the model

```bash
python -m model_pipeline.train \
  --config model_pipeline/configs/base.yaml \
  --train-voxel-dir data_output/train_set/voxel \
  --train-label-dir data_output/train_set/label \
  --val-voxel-dir data_output/val_set/voxel \
  --val-label-dir data_output/val_set/label \
  --epw epw/SWE_VG_Goteborg.City.AP.025120_TMYx.2011-2025.epw \
  --output-dir /path/to/new-training-output
```

Training combines the whole-building sparse spatial representation, target-zone and neighboring-zone context, and known weather and operating conditions to predict zone-level temperature trajectories. Hyperparameters and training options are defined in [`model_pipeline/configs/base.yaml`](model_pipeline/configs/base.yaml).

For a short functional training check using the included cases, replace the configuration with `model_pipeline/configs/main_smoke.yaml` and add `--epochs 1 --max-steps 1`. Write test checkpoints to a separate temporary directory.

### 4. Run temperature prediction

```bash
python -m inference \
  --checkpoint checkpoints/main/best_model.pt \
  --voxel-dir data_output/test_set/voxel \
  --label-dir data_output/test_set/label \
  --epw epw/SWE_VG_Goteborg.City.AP.025120_TMYx.2011-2025.epw \
  --start 07-18 --hours 168 \
  --output-dir data_output/results
```

The default example predicts a single **168-hour window beginning July 18**, with a preceding 72-hour warm-up. Use `--start` to choose another start date and `--hours` to set the forecast duration. `--building-id` selects an individual building. The checkpoint's normalization and run configuration are loaded from its associated files; custom paths can be supplied using `--norm` and `--config`.

The model uses building features, weather, and known operating conditions for prediction. Reference operative temperatures in the label files are used for comparison, not as observed indoor-temperature history fed into the forecast.

Each building produces one Excel workbook in `data_output/results/`, with one row per zone and forecast hour. The `Predictions` sheet contains:

| Column | Description |
| --- | --- |
| `building_id` | Building identifier |
| `zone_id` | Zone identifier |
| `timestamp` | Month-day and hour (`MM-DD HH:MM`) |
| `q10_C` | 0.1 quantile of predicted operative temperature (°C) |
| `q50_C` | Median predicted operative temperature (°C) |
| `q90_C` | 0.9 quantile of predicted operative temperature (°C) |
| `reference_op_C` | EnergyPlus reference operative temperature (°C) |
| `outdoor_temperature_C` | Outdoor dry-bulb temperature (°C) |

Timestamps omit the year because the recommended TMY weather file combines representative months from multiple years.

## Included building cases

The 20 cases are organized as follows:

| Split | Seeds | Buildings |
| --- | --- | ---: |
| Training | 101–109 | 9 |
| Validation | 110 | 1 |
| Test | 111–120 | 10 |
| **Total** | **101–120** | **20** |

`data_output/manifest.json` maps seeds to building IDs, file names, and dataset splits. The repository contains example generated data and the corresponding prediction spreadsheets for inspection and exploration. These cases can also be used to exercise the individual modules.

## Model and configuration

The model architecture and training implementation are located in `model_pipeline/`, with configuration profiles in `model_pipeline/configs/`. The selected trained model is stored in `checkpoints/main/`, together with its associated normalization and provenance information.

For the full model and checkpoint provenance, refer to `checkpoints/main/provenance.json`.


## License

Copyright (c) 2026 Chengrui Yang.

The project materials are provided under the following licenses:

| Material | Scope | License |
| --- | --- | --- |
| Original project code and documentation | Source modules, configuration files, and documentation | [MIT License](LICENSE) |
| MAIN model weights and associated files | `checkpoints/main/`, including `best_model.pt` and normalization statistics | [MIT License](LICENSE) |
| Generated example data and results | `data_output/`: HBJSON, metadata, geometry plots, voxel NPZ, EnergyPlus label NPZ/XLSX, prediction XLSX, and the split manifest | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

When reusing the example datasets, please credit Chengrui Yang (2026), *Geometry-Aware Zone-Level Thermal Prediction*, provide a link to this repository and the CC BY 4.0 license, and indicate any modifications.

Third-party software dependencies remain subject to their respective licenses, including AGPL-3.0 for certain Honeybee/Ladybug packages. The EPW weather files are not included in this repository and are not covered by the project licenses. See [epw/README.md](epw/README.md) for weather data sources and download instructions.
