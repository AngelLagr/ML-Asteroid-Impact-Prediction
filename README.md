# Asteroid Impact Prediction with Machine Learning

## Description  
As many of you have probably heard, the asteroid 2024 YR4 (YRA) has been getting some serious attention lately, with a reported 3% chance of colliding with Earth. Naturally, I couldn’t resist diving into this topic myself and figuring out if we should really be worried.

So, I set out to see if I could use machine learning to predict whether asteroids like YRA are truly dangerous or just hyped up. The goal? To create a model that can classify asteroids based on their orbital parameters (like diameter, eccentricity, etc.) and determine if they pose a threat to our planet.

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
Ok, so here's where it gets interesting. We ran the model with all the fancy preprocessing, and let me tell you – we were seriously on the edge of our seats here.

But guess what? The results came in:

- Accuracy: 0.9999 😲 (it's almost too good to be true!)
- Precision: 0.9766
- Recall: 0.9843
- F1 Score: 0.9804

And the confusion matrix:
```bash
[[59870     3]
 [    2   125]]
```
We threw in some testing with a really threatening asteroid we've seen from the past and the infamous 2024 YR4 (yeah, that one everyone's been talking about).

For the dangerous asteroid: BAM! It was correctly flagged as Dangerous.
For YRA 2024? The model calmly predicted it as Non-Dangerous. Phew! 😅

So, no need to start building your fallout shelters just yet! We've got the cosmic threat under control... for now.

---

## Disclaimer
Now, let's not get carried away here – I'm not a professional astronomer or cosmic danger expert, just an engineering student dabbling with machine learning. While the model's results look pretty solid, nothing here is really 100% accurate. So, unfortunately, we’ll still have to wait a bit before we can say for sure if we’re a 100% safe or not :( !
