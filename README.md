# Blood Group Detection Using Fingerprints

This project implements a **deep learning model to predict human blood groups from fingerprint images** using a **ResNet-based Convolutional Neural Network (CNN)**. The trained model analyzes fingerprint patterns and classifies them into different blood group categories.

The model is trained using **TensorFlow/Keras** and saved as **`model_best.h5`**, which is used for prediction in the application.

---

## Technologies Used

**Python**

* Used for implementing the machine learning pipeline and application.

**TensorFlow / Keras**

* Used to build, train, and save the deep learning model.

**ResNet (Residual Neural Network)**

* A deep CNN architecture used for extracting fingerprint features and performing classification.

**Computer Vision**

* Used to process and analyze fingerprint images before feeding them to the model.

---

## Blood Group Classes

The model predicts the following blood groups:

* A+
* A-
* B+
* B-
* AB+
* AB-
* O+
* O-

---

## How It Works

1. Upload a fingerprint image.
2. Image is preprocessed and resized.
3. The **ResNet model** extracts features from the fingerprint.
4. The model predicts the corresponding **blood group**.

---

## Model File

```
model_best.h5
```

This file contains the **trained ResNet model used for prediction**.

---

## Project Purpose

The goal of this project is to explore the **use of deep learning and fingerprint patterns for blood group classification** using computer vision techniques.

---

## Screenshots

Project screenshots demonstrating the interface and prediction results are included in this repository.

## Screenshots

### Application Interface
![Interface](/1st.png)
![img](/2nd.png)
![img](/3rd.png)
![img](/4rth.png)
