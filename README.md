# Asteroid Impact Prediction with Machine Learning

## Description  
This project leverages **machine learning** techniques to predict whether an asteroid is dangerous or not, based on its orbital parameters.  
The goal is to train a classifier to determine the **dangerousness** of asteroids using features such as **diameter, eccentricity, and orbit-related values**.

The model is trained using a **Random Forest Classifier**, and the dataset is preprocessed with **SMOTE** (Synthetic Minority Over-sampling Technique) for balancing the classes and **RandomUnderSampler** for under-sampling.

**Current version:** The model is trained and evaluated using a dataset with asteroid characteristics and their classification (dangerous or not).  
**Final goal:** To generalize the model to handle new incoming data from astronomical observations, enabling real-time predictions of asteroid threats.

---

## Project Structure  
- **`train_model.py`**: Contains the training pipeline with data preprocessing and model training.  
- **`trained_model.joblib`**: File storing the trained model.  
- **`load_model.py`**: Script to load the trained model and run predictions on new data.  
- **`dataset.csv`**: Example dataset used for training the model.  
- **`performance_metrics.py`**: Script that outputs the model's evaluation metrics, such as accuracy, precision, recall, and F1 score.

---

## Final Objective  
The ultimate goal is to develop a system capable of predicting asteroid impacts with high accuracy, using available orbital parameters. Initially, the focus is on predicting asteroids' dangerousness based on historical data, but the vision is to integrate this model into an automated monitoring system for real-time asteroid detection and risk assessment.

---

## Technologies Used  
- **Python**  
- **scikit-learn** (machine learning models and preprocessing)  
- **imblearn** (SMOTE and RandomUnderSampler)  
- **joblib** (model persistence)  
- **pandas** (data manipulation)  
- **matplotlib** (performance visualization)

---

## Execution   
```bash
python train_and_try_model.py
```
---
## Results
I succeeded to obtain these metrics :
- Accuracy: 0.9999
- Precision: 0.9766
- Recall: 0.9843
- F1 Score: 0.9804

And this confusion matrix :
[[59870     3]
 [    2   125]]


