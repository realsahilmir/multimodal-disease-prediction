# 🩺 Multimodal Disease Prediction Web Application

## 📌 Project Overview

The **SUS DISEASE DETECTOR** is a machine learning–based web application designed to assist in **early disease screening** using medical data and medical images.

The platform allows users to enter health parameters or upload medical images. The system then analyzes the data using **machine learning and deep learning models** to estimate the risk of various diseases.

The goal of this project is to demonstrate how **Artificial Intelligence can assist healthcare systems** by providing quick preliminary analysis and encouraging early medical consultation.

---

# 🎯 Problem Statement

Healthcare diagnosis can be time-consuming and often requires multiple tests and expert analysis. Many people delay medical consultation due to lack of awareness or accessibility.

This project aims to build a **machine learning powered web application** that helps users perform **early disease screening** by analyzing their medical data.

The application allows users to:

* Upload or enter medical information
* Get AI-based disease risk predictions
* Communicate with doctors through email
* Book medical appointments through the platform

This system acts as a **support tool for early detection and awareness**, not as a replacement for professional medical diagnosis.

---

# 💡 Why This Project?

Human diagnosis is highly skilled but still prone to error or delay. Machine learning models can analyze patterns in medical datasets and provide **fast and consistent predictions**.

While researching datasets, we explored different medical systems including:

* Allopathic Medicine
* Homeopathy
* Ayurvedic Medicine

However, due to the **limited availability of structured datasets** in homeopathy and ayurvedic medicine, we focused on **allopathic medical datasets** that are publicly available from platforms such as Kaggle and the UCI Machine Learning Repository.

These datasets contain structured patient data and medical images suitable for training machine learning models.

---

# 🧠 Multimodal AI Approach

This project uses a **multimodal machine learning approach**, meaning the system processes **different types of medical data**.

### 1️⃣ Tabular Medical Data

Used for diseases that rely on patient health parameters such as:

* Age
* Blood pressure
* Glucose level
* BMI
* Cholesterol
* Lifestyle indicators

These datasets are processed using **traditional machine learning models**.

---

### 2️⃣ Medical Image Data

Some diseases require image analysis.

Examples include:

* Pneumonia detection from chest X-rays
* Malaria detection from microscopic blood cell images

For these tasks, **Convolutional Neural Networks (CNN)** are used.

---

# ⚙️ System Architecture

Frontend Interface (User Input)
⬇
Flask Web Application Backend
⬇
Machine Learning / Deep Learning Model
⬇
Prediction Output
⬇
Doctor Consultation / Appointment Option

---

# 📂 Project Directory Structure

```
├── Python Notebooks
│   └── Model training notebooks
│
├── trained_models
│   └── .pkl model files
│
├── static
│   ├── logos
│   ├── css
│   ├── js
│   ├── images
│   └── fonts
│
├── templates
│   ├── index.html
│   ├── home.html
│   ├── about.html
│   ├── contact.html
│   ├── services.html
│   └── disease prediction pages
│
├── app.py
├── requirements.txt
├── runtime.txt
└── README.md
```

---

# 🚀 Quick Start Guide

### Step 1 — Clone the Repository

```
git clone <repository-link>
```

---

### Step 2 — Install Dependencies

Navigate to the project folder and run:

```
pip install -r requirements.txt
```

---

### Step 3 — Run the Application

```
python app.py
```

or

```
flask run
```

---

### Step 4 — Open the Web Application

Open your browser and go to:

```
http://127.0.0.1:5000/
```

---

# 🧪 Diseases Predicted

The system currently supports prediction for the following diseases:

* Diabetes
* Breast Cancer
* Heart Disease
* Kidney Disease
* Liver Disease
* Malaria
* Pneumonia

---

# 🤖 Machine Learning Models

| Disease        | Model Type          | Accuracy |
| -------------- | ------------------- | -------- |
| Diabetes       | Machine Learning    | 98.25%   |
| Breast Cancer  | Machine Learning    | 98.25%   |
| Heart Disease  | Machine Learning    | 85.25%   |
| Kidney Disease | Machine Learning    | 99%      |
| Liver Disease  | Machine Learning    | 78%      |
| Malaria        | Deep Learning (CNN) | 96%      |
| Pneumonia      | Deep Learning (CNN) | 95%      |

---

# 📊 Dataset Sources

All datasets used in this project were obtained from **Kaggle** and **UCI Machine Learning Repository**.

Datasets used:

* Diabetes Dataset
* Breast Cancer Dataset
* Heart Disease Dataset
* Chronic Kidney Disease Dataset
* Liver Disease Dataset
* Malaria Cell Image Dataset
* Pneumonia Chest X-ray Dataset

These datasets contain **thousands of medical records and images used for training predictive models**.

---

# 🛠 Technologies Used

### Programming

* Python

### Machine Learning

* Scikit-learn
* TensorFlow
* Keras

### Backend Framework

* Flask

### Frontend

* HTML
* CSS
* JavaScript

### Model Storage

* Pickle (.pkl)

---

# 🔮 Future Improvements

Possible enhancements for future versions:

* Integration with wearable health devices
* Real-time health monitoring
* Mobile application development
* Explainable AI for transparent predictions
* Integration with hospital databases
* Support for additional diseases

---

# ⚠️ Disclaimer

This project is created for **educational and research purposes only**.

The predictions generated by this system are **not a substitute for professional medical advice, diagnosis, or treatment**. Always consult a qualified healthcare professional for medical concerns.

---
