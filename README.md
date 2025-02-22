# Asteroid Impact Prediction with Machine Learning

## Description  
This project leverages **machine learning** techniques to predict whether an asteroid is dangerous or not, based on its orbital parameters.  
The goal is to train a classifier to determine the **dangerousness** of asteroids using features such as **diameter, eccentricity, and orbit-related values**.

The model is trained using a **Random Forest Classifier**, and the dataset is preprocessed with **SMOTE** (Synthetic Minority Over-sampling Technique) for balancing the classes and **RandomUnderSampler** for under-sampling.

---

## Project Structure  
- **`train_and_try_model.py`**: Contains the whole code to generate the model and test it   
- **`dataset.csv`**: Example dataset used for training the model that I have founded on Kaggle

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


