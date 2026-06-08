# Gaze Estimation using Convolutional Neural Networks

## 📌 Project Overview
This repository contains a PyTorch-based Deep Learning pipeline designed for **Gaze Estimation**. The model predicts the 2D gaze direction (yaw and pitch) of a human eye using cropped eye images and 3D head pose data. 

## 📊 Dataset
The model is trained and evaluated on the **[MPIIGaze Dataset](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild/)**, a standard benchmark for appearance-based gaze estimation in the wild.
* **Input:** Normalized grayscale eye images and a 3D head pose vector.
* **Target:** 2D gaze vectors (yaw and pitch in radians).

## 🧠 Model Architecture
The project uses a custom Convolutional Neural Network (CNN) named `GazeEstimationNet`. It works by:
1. Extracting visual features from the eye images using convolutional layers.
2. Processing the 3D head pose data through fully connected layers.
3. Fusing both feature sets to predict the final continuous gaze coordinates.

## ⚙️ Training & Evaluation
* The model was built and trained using **PyTorch**.
* The training pipeline includes data augmentation to improve model robustness.
* The evaluation was conducted using **5-Fold Cross Validation**, achieving an average Mean Angular Error of approximately 5.5°.

## 🚀 How to Run
This project was primarily developed in Google Colab. The dataset is downloaded automatically via `kagglehub`.

1. Clone this repository:
   ```bash
   git clone [https://github.com/khrystynakoblan/GazeEstimation.git](https://github.com/khrystynakoblan/GazeEstimation.git)
   cd GazeEstimation.
2. Open the gaze_estimation_main_code.ipynb file in Google Colab.
3. Run the cells sequentially to download the data, pre-process it, build the model, and start the training process.
