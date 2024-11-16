import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import joblib
import librosa
import reflex as rx
import matplotlib.pyplot as plt

# Load and preprocess the data
try:
    df = pd.read_csv('Hackathon_Project\models\parkinsons_data.csv')
except FileNotFoundError:
    print("File not found. Please check the file path.")
    exit()

# Drop unnecessary columns and handle missing values
df = df.drop(columns=['name'])  # Drop 'name' column as it is not a feature
df = df.fillna(df.mean(numeric_only=True))  # Fill missing values with column means

# Select relevant columns for training
selected_columns = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 
    'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 
    'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 
    'RPDE', 'DFA', 'spread1', 'spread2'
]

# Separate features and target variable
X = df[selected_columns]
y = df['status']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler.pkl')  # Save the scaler for future use

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Hyperparameter tuning for Random Forest
param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_rf_model = grid_search.best_estimator_
joblib.dump(best_rf_model, 'best_rf_model.pkl')  # Save the model

# Evaluate the model
y_pred = best_rf_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, best_rf_model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()


# Feature extraction for audio files (18 features)
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)

    # Extract fundamental frequency features
    f0, voiced_flag, voiced_probs = librosa.pyin(y, 
                                                 fmin=librosa.note_to_hz('C2'), 
                                                 fmax=librosa.note_to_hz('C7'))
    voiced_flag = voiced_flag.astype(bool)
    f0_voiced = f0[voiced_flag]

    # 1-3: Fundamental frequency statistics
    mdvp_fo = np.nanmean(f0_voiced)
    mdvp_fhi = np.nanmax(f0_voiced)
    mdvp_flo = np.nanmin(f0_voiced)

    # Handle NaN or Inf values
    if np.isnan(mdvp_fo) or np.isnan(mdvp_fhi) or np.isnan(mdvp_flo):
        mdvp_fo, mdvp_fhi, mdvp_flo = 0, 0, 0

    # 4-8: Jitter variations
    jitter_percent = np.nanstd(f0_voiced) / mdvp_fo * 100 if mdvp_fo != 0 else 0
    jitter_abs = np.nanmean(np.abs(np.diff(f0_voiced)))
    jitter_rap = np.nanmean(np.abs(np.diff(f0_voiced, n=2))) / mdvp_fo if mdvp_fo != 0 else 0
    jitter_ppq = np.nanmean(np.abs(np.diff(f0_voiced, n=4))) / mdvp_fo if mdvp_fo != 0 else 0
    jitter_ddp = np.nanmean(np.abs(np.diff(np.diff(f0_voiced)))) / mdvp_fo if mdvp_fo != 0 else 0

    # Handle NaN or Inf values in jitter
    jitter_percent = 0 if np.isnan(jitter_percent) or np.isinf(jitter_percent) else jitter_percent
    jitter_abs = 0 if np.isnan(jitter_abs) or np.isinf(jitter_abs) else jitter_abs
    jitter_rap = 0 if np.isnan(jitter_rap) or np.isinf(jitter_rap) else jitter_rap
    jitter_ppq = 0 if np.isnan(jitter_ppq) or np.isinf(jitter_ppq) else jitter_ppq
    jitter_ddp = 0 if np.isnan(jitter_ddp) or np.isinf(jitter_ddp) else jitter_ddp

    # 9-14: Shimmer variations
    rms = librosa.feature.rms(y=y)[0]
    if rms.size == 0 or np.any(np.isnan(rms)) or np.any(np.isinf(rms)):
        rms = np.zeros_like(rms)  # Replace invalid RMS with zeros
    shimmer = np.nanstd(rms) / np.nanmean(rms) * 100 if np.nanmean(rms) != 0 else 0
    shimmer_db = 20 * np.log10(np.nanmax(rms) / np.nanmin(rms)) if np.nanmin(rms) != 0 else 0
    shimmer_apq3 = np.nanmean(np.abs(np.diff(rms, n=2))) / np.nanmean(rms) if np.nanmean(rms) != 0 else 0
    shimmer_apq5 = np.nanmean(np.abs(np.diff(rms, n=4))) / np.nanmean(rms) if np.nanmean(rms) != 0 else 0
    shimmer_apq = np.nanmean(np.abs(np.diff(rms, n=6))) / np.nanmean(rms) if np.nanmean(rms) != 0 else 0
    shimmer_dda = np.nanmean(np.abs(np.diff(np.diff(rms)))) / np.nanmean(rms) if np.nanmean(rms) != 0 else 0

    # Handle NaN or Inf values in shimmer
    shimmer = 0 if np.isnan(shimmer) or np.isinf(shimmer) else shimmer
    shimmer_db = 0 if np.isnan(shimmer_db) or np.isinf(shimmer_db) else shimmer_db
    shimmer_apq3 = 0 if np.isnan(shimmer_apq3) or np.isinf(shimmer_apq3) else shimmer_apq3
    shimmer_apq5 = 0 if np.isnan(shimmer_apq5) or np.isinf(shimmer_apq5) else shimmer_apq5
    shimmer_apq = 0 if np.isnan(shimmer_apq) or np.isinf(shimmer_apq) else shimmer_apq
    shimmer_dda = 0 if np.isnan(shimmer_dda) or np.isinf(shimmer_dda) else shimmer_dda

    # 15-16: Noise ratios
    harmonic = librosa.effects.harmonic(y)
    percussive = librosa.effects.percussive(y)
    nhr = np.nanmean(percussive) / np.nanmean(harmonic) if np.nanmean(harmonic) != 0 else 0
    hnr = np.nanmean(harmonic) / np.nanmean(percussive) if np.nanmean(percussive) != 0 else 0

    # 17: RPDE (using zero crossing rate as proxy)
    rpde = np.mean(librosa.feature.zero_crossing_rate(y))

    # 18: DFA (using spectral rolloff as proxy)
    dfa = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

    # 19-20: Frequency variation spreads (using spectral bandwidth and contrast)
    spread1 = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spread2 = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))

    features = np.array([
        mdvp_fo, mdvp_fhi, mdvp_flo,
        jitter_percent, jitter_abs, jitter_rap, jitter_ppq, jitter_ddp,
        shimmer, shimmer_db, shimmer_apq3, shimmer_apq5, shimmer_apq, shimmer_dda,
        nhr, hnr, rpde, dfa, spread1, spread2
    ])
    
    features = np.nan_to_num(features)  # Replace NaN and inf with 0
    features[np.isinf(features)] = 0  # Handle infinity
    return features

