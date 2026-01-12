## Megajánlott jegyért

## 👥 Team Information
# Team name: Overfitters United
| Team Member | Neptun Code |
|--------------|-------------|
| Gergő Czakó |VCXUNF |
| Sági Benedek | ECSGGY |

The third member left us, so only the two of us worked on this project.


## 📘 Project Overview

This project uses the **Airbus Ship Detection Challenge** dataset from *Kaggle*,  
with the goal of identifying and segmenting ships in satellite images.

# Airbus Ship Detection – Milestone 1
**Focus:** Data source identification and download scripts.

## Data Source
- Kaggle: **Airbus Ship Detection Challenge**(https://www.kaggle.com/competitions/airbus-ship-detection)
  - Folders: `train_v2/`, `test_v2/`
  - Annotation file: `train_ship_segmentations_v2.csv`

**Download scripts included in this repository:**
| File | Description |
|------|--------------|
| `download_airbus_kaggle.ipynb` | Google Colab notebook – Kaggle API setup, dataset download, unzip, and verification |

## 🧠 Training, Data Loading & Evaluation

The full workflow for the project — including:

- custom Keras `Sequence` generator for data loading  
- balanced train/validation split  
- UNet model building  
- training setup with callbacks (checkpointing, LR scheduling, early stopping)  
- evaluation metrics (IoU, Dice, F2 pixel score)  
- visualization of masks and predictions  

is implemented in the following notebook:

| File | Description |
|------|-------------|
| `airbus_training_pipeline.ipynb` | Complete data preparation, training, and evaluation pipeline |

### Build and Run (Docker)

This project provides a GPU-enabled Docker image based on
`tensorflow/tensorflow:2.15.0-gpu`. The container runs the full pipeline
by executing `run.sh` on startup.

The container executes:
- data loading and preprocessing
- model training
- validation and evaluation

Before running the container, ensure that:
- Docker and Docker Compose are installed
- the Airbus Ship Detection dataset is downloaded locally

---

#### Linux / macOS

```bash
HOST_DATA_DIR=/absolute/path/to/your/local/data \
docker compose up --build > log/run.log 2>&1
```

#### Windows (PowerShell)

```powershell
$env:HOST_DATA_DIR="C:/absolute/path/to/your/local/data"
docker compose up --build > log/run.log 2>&1
```

---

**Notes:**
- Replace `HOST_DATA_DIR` with the folder that contains the Airbus dataset
  (`train_v2/`, `test_v2/`, and `train_ship_segmentations_v2.csv`).
- The dataset directory is mounted into the container at `/work/data`.
- Logs are captured into `run.log` in the repository root using output redirection.

## 📁 File Structure

The repository is organized as follows:

```
.
├── notebook/
│   ├── download_airbus_kaggle.ipynb
│   └── airbus_training_pipeline.ipynb
├── src/
│   └── helper modules and reusable components
├── documentation/
│   └── report and assignment-related files
├── log/
│   └── run.log
├── Dockerfile
├── docker-compose.yaml
├── requirements.txt
├── run.sh
└── README.md
```

### Key Components

- **`notebook/`**
  - `download_airbus_kaggle.ipynb`: Notebook for downloading and verifying
    the Airbus Ship Detection dataset using the Kaggle API
  - `airbus_training_pipeline.ipynb`: End-to-end pipeline for data
    preparation, model training, evaluation, and visualization

- **`src/`**
  - Helper modules and reusable components used by the training pipeline
    (e.g. data loading utilities, model-related functions)

- **`log/`**
  - Includes the running logs that describes the result (output/console) of the docker app.

- **Root directory**
  - `Dockerfile`: Defines the GPU-enabled training environment and
    installs all required dependencies
  - `docker-compose.yaml`: Defines the `segmentai` service with NVIDIA
    GPU support
  - `requirements.txt`: Python dependencies required for training and
    evaluation
  - `run.sh`: Entry-point script executed when the Docker container
    starts, responsible for launching the training pipeline
