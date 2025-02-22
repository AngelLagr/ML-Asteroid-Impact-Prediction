# Asteroid Impact Prediction with Machine Learning

## Description  
This project leverages **machine learning** techniques to predict whether an asteroid is dangerous or not for Earth, based on its orbital parameters.  

The goal is to train a classifier to determine the **dangerousness** of asteroids using features such as **diameter, eccentricity, and orbit-related values**.

The model is trained using a **Random Forest Classifier**, and the dataset is preprocessed with **SMOTE** (Synthetic Minority Over-sampling Technique) for balancing the classes and **RandomUnderSampler** for under-sampling.

---

## Project Structure  
- **`train_and_try_model.py`**: Contains the whole code to generate the model and test it but you'll need to download the database (look below)

---

## Technologies Used  
- **Python 3.11**  
- **scikit-learn**
- **imblearn** (SMOTE and RandomUnderSampler)  
- **joblib**
- **pandas**  

---
## Download the database
**Either you download it directly on kaggle :** https://www.kaggle.com/datasets/sakhawat18/asteroid-dataset/data

**Either you use their library on python :**

First execute this command for your conda env :
```bash
pip install kagglehub[pandas-datasets]
```
And add these codelines to the train_and_try_model.py (that will replace the data = pd.read_csv(...))
```bash
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
file_path = ""

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "sakhawat18/asteroid-dataset",
  file_path,
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)
```

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


