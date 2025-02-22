import pandas as pd
import joblib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix

# List of interesting features
features = [
    "H","neo", "diameter", "albedo", "e", "a", "q", "i", "om", "w", "ma",
    "ad", "n", "tp", "per", "per_y", "moid", "moid_ld", "sigma_e", "sigma_a",
    "sigma_q", "sigma_i", "sigma_om", "sigma_w", "sigma_ma", "sigma_ad",
    "sigma_n", "sigma_tp", "sigma_per", "rms"
]

# Final label to interpret the model's result
output = ["Non-dangerous", "Dangerous"]

# Loading data
print("🔄 Loading the dataset...")
data = pd.read_csv("dataset.csv", low_memory=False)
print(f"Data loaded ({data.shape[0]} rows, {data.shape[1]} columns) \n")

# Dropping rows with NaN in `pha`
data = data.dropna(subset=["pha"])

# Creating a random sample of the DataFrame
print("🔄 Sampling and preparing data")
sample_size = 200000  # We don't want too much data to avoid long processing time
data = data.sample(n=sample_size)

# Filling y
y = data["pha"]

# Transforming the "pha" column to numerical values in y
if y.dtype == 'O':
    y = y.map({"Y": 1, "N": 0})
    
# Transforming the "neo" column to numerical values
data["neo"] = data["neo"].map({"Y": 1, "N": 0})

# Filling X
X = data[features]

# Replacing NaN in X with the mean
X = X.fillna(X.mean())

# Keeping 30% of data for validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(f"Class distribution in training set before data-processing': {y_train.value_counts()} \n")

      # --- SMOTE ---
print("🔄 Using SMOTE to balance the classes...")
smote = SMOTE(sampling_strategy=0.5)  # SMOTE is used to oversample the minority class (choosing 0.5 to make it 50% of the majority class size)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train) 
print(f"SMOTE completed. New dataset: Class distribution in training set after SMOTE': {y_train_res.value_counts()} \n")

# --- Under-sampling ---
print("🔄 Under-sampling data with RandomUnderSampler...")
rus = RandomUnderSampler()
X_train_res_rus, y_train_res_rus = rus.fit_resample(X_train_res, y_train_res)
print(f"Under-sampling completed. Class distribution in training set after Under-Sampling': {y_train_res_rus.value_counts()} \n")

# --- Model Training ---
print("🔄 Training the model...")
model = RandomForestClassifier(n_estimators=30, verbose=1)

# Adding a progress bar during training
for _ in tqdm(range(1), desc="Training in progress"):
    model.fit(X_train_res_rus, y_train_res_rus)

print("Model successfully trained! \n")

# --- Model Evaluation ---
print("✅ Results:")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")    
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# --- Saving the model ---
joblib.dump(model, "trained_model.joblib")
print("\n💾 Model saved in 'trained_model.joblib'. \n")

# --- Model Loading Function ---
def load_model(model_path="trained_model.joblib"):
    """Loads and returns the trained model."""
    return joblib.load(model_path)

# --- Testing on YRA 2024 ---
YRA_2024_features = {
    "H": 23.96,
    "neo" : 1,
    "diameter": 55,
    "albedo": 0.154,
    "e": 0.66427,
    "a": 2.54001,
    "q": 0.85276,
    "i": 3.45326,
    "om": 271.4123,
    "w": 134.64151,
    "ma": 351.07993,
    "ad": 4.22726,
    "n": 0.24347,
    "tp": 2460637.1368,
    "per": 1478.6017,
    "per_y": 4.0482,
    "moid": 0.00283,
    "moid_ld": 1.10,
    "sigma_e": 0.001,
    "sigma_a": 0.001,
    "sigma_q": 0.001,
    "sigma_i": 0.001,
    "sigma_om": 0.001,
    "sigma_w": 0.001,
    "sigma_ma": 0.001,
    "sigma_ad": 0.001,
    "sigma_n": 0.001,
    "sigma_tp": 0.001,
    "sigma_per": 0.001,
    "rms": 0.40
}

# --- Test on a dangerous asteroid ---
dangerous_asteroid_features = {
    "H": 15.3,  # Absolute magnitude
    "neo":1, # NEO label
    "diameter": 2.56,  # Diameter
    "albedo": 0.290,  # Albedo
    "e": 0.15,  # Eccentricity
    "a": 1.245666863784177,  # Semi-major axis
    "q": 0.8278019326218434,  # Perihelion
    "i": 13.3370434846758,  # Inclination
    "om": 337.1869752155534,  # Longitude of the ascending node
    "w": 276.956067645158,  # Argument of perihelion
    "ma": 235.1126416339228,  # Mean anomaly
    "ad": 1.66353179494651,  # Aphelion distance
    "n": 0.7089264923814854,  # Orbital speed
    "tp": 2459176.664044803214,  # Time of perihelion passage
    "per": 20201123.1640448,  # Orbital period
    "per_y": 11.812671595,  # Period in years
    "moid": 1.1323E-8,  # Minimum impact distance
    "moid_ld": 1.5884E-10,  # Minimum impact distance in LD units
    "sigma_e": 1.4171E-8,  # Error on eccentricity
    "sigma_a": 2.5705E-6,  # Error on semi-major axis
    "sigma_q": 2.6329E-6,  # Error on perihelion
    "sigma_i": 3.4911E-6,  # Error on inclination
    "sigma_om": 3.0666E-6,  # Error on longitude of ascending node
    "sigma_w": 2.1212E-10,  # Error on argument of perihelion
    "sigma_ma": 1.3559E-10,  # Error on mean anomaly
    "sigma_ad": 4.3566E-6,  # Error on aphelion distance
    "sigma_n": 9.7128E-8,  # Error on orbital speed
    "sigma_tp": 0.40639,  # Error on perihelion passage time
    "sigma_per": 0.40639,  # Error on orbital period
    "rms": 0.40639  # Error on trajectory
}

# Conversion to DataFrame
dangerous_asteroid_df = pd.DataFrame([dangerous_asteroid_features])
YRA_2024_df = pd.DataFrame([YRA_2024_features])

# Loading the model and prediction
loaded_model = load_model()

print("📌 Tests:")
prediction = loaded_model.predict(dangerous_asteroid_df)[0]
print("Prediction for a dangerous asteroid:", output[prediction])

prediction = loaded_model.predict(YRA_2024_df)[0]
print("Prediction for YRA 2024 asteroid:", output[prediction])
