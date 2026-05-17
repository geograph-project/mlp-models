# Geograph ML Models & PHP Inference Port

This repository is a consolidated collection of standard machine learning models, providing a unified interface for inference 
across both Python (PyTorch) and native PHP runtimes.

All code in this repository is released under the **MIT License**.

---

## Architecture Overview

The core goal of this repository is model standardization. Each distinct model architecture has been normalized into a 
consistent `predict_api($inputs)` structure.

* **Training Scripts & Datasets:** The original PyTorch training code and datasets are hosted separately (available on Kaggle 
and GitHub via the links below). They are not included in this repository.
* **Model Weights:** Raw `.pth` checkpoints and weights are not distributed here and must be acquired separately.

---

## Python Framework (PyTorch)

The Python framework provides the baseline PyTorch implementation of the standardized prediction API.

### Weight Export & JSON Generation
To bridge the gap between Python and PHP, the framework includes utilities to parse PyTorch checkpoints and export them as JSON 
weights.

* Run the `export_models.py` script to generate a `weights.json` file for each architecture. 
* **Fallback Behavior:** If no `.pth` checkpoint is found for a model, the script will automatically initialize a checkpoint 
with **random weights** and export it. *(Note: Random weights will not produce valid analytical predictions).*

---

## PHP Port (Native Inference Engine)

We have implemented a native PHP port of the inference architecture (`GeographModelBase`). 

Because these models accept precomputed data (such as image and text CLIP embeddings) rather than raw media assets, running raw 
mathematical inference directly inside a standard web runtime becomes entirely feasible.

### Notes & Testing

* **Performance:** While fully functional for production environments with low-frequency demands, a native PHP matrix layout is 
significantly slower than hardware-accelerated Python.

* **Dry Run:** An `example.json` file containing precomputed embeddings for a single image is provided. To get authentic 
predictions, ensure you have ran the Python export script to generate valid `weights.json` data files for your models to read.

---

## Links & Resources

While not all models have external repositories available yet, you can find the open-source training configurations, specific code, and reference datasets for the primary models below:

### 1. Clip / Landscape Model
* **Source Code & Training:** [SpaceTimeLab/ClipTheLandscape (GitHub)](https://github.com/SpaceTimeLab/ClipTheLandscape)
* **Dataset:** [Predict Geographic Context from Landscape Photos (Kaggle Competition)](https://www.kaggle.com/competitions/predict-geographic-context-from-landscape-photos)

### 2. Types Model
* **Source Code, Training, Weights:** [geograph-project/geograph-type-classifier (GitHub)](https://github.com/geograph-project/geograph-type-classifier)
* **Dataset:** [Geograph Types Dataset 1 (Kaggle)](https://www.kaggle.com/datasets/barrybhunter/geograph-types-dataset-1/)

### 3. Subject Model
* **Source Code & Training:** [geograph-project/geograph-subject-classifer (GitHub)](https://github.com/geograph-project/geograph-subject-classifer)
* **Dataset:** [Geograph Subject Dataset 1 (Kaggle)](https://www.kaggle.com/datasets/barrybhunter/geograph-subject-dataset-1/)
* **Weights:** [Geograph Subject Model (Kaggle)](https://www.kaggle.com/models/barrybhunter/geograph-subject-model)


