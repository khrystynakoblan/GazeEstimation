# Gaze Estimation

## Project Overview
This repository contains a PyTorch-based Deep Learning pipeline designed for **Gaze Estimation**. The model predicts the 2D gaze direction (yaw and pitch) of a human eye using cropped eye images and 3D head pose data. 

## Dataset
The model is trained and evaluated on the **[MPIIGaze Dataset](https://www.kaggle.com/datasets/dhruv413/mpiigaze)**, a standard benchmark for appearance-based gaze estimation in the wild.
* **Input:** Normalized grayscale eye images and a 3D head pose vector.
* **Target:** 2D gaze vectors (yaw and pitch in radians).

## Model Architecture
The project uses a custom Convolutional Neural Network (CNN) named `GazeEstimationNet`. It works by:
1. Extracting visual features from the eye images using convolutional layers.
2. Processing the 3D head pose data through fully connected layers.
3. Fusing both feature sets to predict the final continuous gaze coordinates.

## Training & Evaluation
* The model was built and trained using **PyTorch**.
* The training pipeline includes data augmentation to improve model robustness.
* The evaluation was conducted using **5-Fold Cross Validation**, achieving an average Mean Angular Error of approximately 5.5°.

## How to Run

1. Upload `source/` folder to your Google Drive
2. Open `main.ipynb` in Google Colab, enable GPU  
   (`Runtime → Change runtime type → T4 GPU`)
3. In **Cell 1**, set `PROJECT_PATH` to the folder where you uploaded the files:
```python
   PROJECT_PATH = '/content/drive/MyDrive/your_folder/source'
```
4. Run all cells sequentially
