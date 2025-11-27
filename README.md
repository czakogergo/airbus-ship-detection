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
