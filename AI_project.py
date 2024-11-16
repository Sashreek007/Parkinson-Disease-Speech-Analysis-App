# Import required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.ensemble import VotingClassifier, StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import joblib
import librosa
import tkinter as tk
from tkinter import filedialog
import numpy as np
import os
from flask import Flask, request, render_template_string
import librosa
import numpy as np
import joblib
import reflex as rx


# Load and preprocess the data
try:
    df = pd.read_csv('parkinsons data.csv')
except FileNotFoundError:
    print("File not found. Please check the file path.")
    exit()

# Drop unnecessary columns and handle missing values
df = df.drop(columns=['name'])  # Drop 'name' column as it is not a feature
df = df.fillna(df.mean(numeric_only=True))  # Fill missing values with column means

# Check for class imbalance
print("Class distribution in target variable:")
print(df['status'].value_counts())

# Define feature patterns
patterns = ['MDVP', 'Shimmer', 'Jitter', 'HNR', 'RPDE', 'DFA', 'spread', 'NHR']

# Select columns based on patterns
selected_columns = [col for col in df.columns if any(pattern in col for pattern in patterns)]
selected_columns.append('status')  # Ensure 'status' is included as the target

# Split the data into features and target
X = df[selected_columns[:-1]]
y = df['status']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler for future use
joblib.dump(scaler, 'scaler.pkl')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define models
voting_model = VotingClassifier(estimators=[
    ('logreg', LogisticRegression()), 
    ('rf', RandomForestClassifier()), 
    ('svc', SVC(probability=True))
], voting='soft')

stacking_model = StackingClassifier(estimators=[
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('svc', SVC(probability=True)),
    ('logreg', LogisticRegression())
], final_estimator=LogisticRegression())

rf_model = RandomForestClassifier(n_estimators=100)
gb_model = GradientBoostingClassifier(n_estimators=100)

# Train and evaluate models
models = {
    'voting': voting_model,
    'stacking': stacking_model,
    'random_forest': rf_model,
    'gradient_boosting': gb_model
}

for name, model in models.items():
    print(f"\nTraining {name} model...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"Results for {name}:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")

# Perform hyperparameter tuning for Random Forest as an example
param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("\nBest parameters from GridSearchCV:")
print(grid_search.best_params_)

# Evaluate the best Random Forest model
best_rf_model = grid_search.best_estimator_
y_pred = best_rf_model.predict(X_test)

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, best_rf_model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Cross-validation scores
cv_scores = cross_val_score(best_rf_model, X_scaled, y, cv=5)
print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f}")

# Save the best model
joblib.dump(best_rf_model, 'best_rf_model.pkl')
print("Model saved as 'best_rf_model.pkl'.")

# Plot learning curve
train_sizes, train_scores, test_scores = learning_curve(best_rf_model, X_scaled, y, cv=5)
plt.plot(train_sizes, train_scores.mean(axis=1), label="Training score")
plt.plot(train_sizes, test_scores.mean(axis=1), label="Test score")
plt.xlabel("Training size")
plt.ylabel("Score")
plt.title("Learning Curves")
plt.legend()
plt.show()

# Feature importance for Random Forest
importances = best_rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

print("\nFeature importances:")
for i in range(X.shape[1]):
    print(f"{X.columns[indices[i]]}: {importances[indices[i]]:.4f}")

# Load the trained model
best_rf_model = joblib.load('best_rf_model.pkl')

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    
    # Extract additional features
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_stft_mean = np.mean(chroma_stft, axis=1)
    
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_contrast_mean = np.mean(spectral_contrast, axis=1)
    
    # Combine all features into a single array
    features = np.hstack([mfccs_mean, chroma_stft_mean, spectral_contrast_mean])
    return features[:20]  # Ensure the number of features matches the model's expectations
'''
def predict_health_status(audio_path, model):
    features = extract_features(audio_path)
    features = features.reshape(1, -1)  # Reshape for the model
    prediction = model.predict(features)
    return "Healthy" if prediction == 1 else "Not Healthy"'''

class State(rx.State):
    """The app state."""
    prediction: str = ""  # Store the prediction result

    @rx.event
    async def handle_upload(self, files: list[rx.UploadFile]):
        """Handle the upload of file(s).

        Args:
            files: The uploaded files.
        """
        for file in files:
            upload_data = await file.read()
            outfile = rx.get_upload_dir() / file.filename

            # Save the file
            with outfile.open("wb") as file_object:
                file_object.write(upload_data)

            # Process the audio file and make prediction
            try:
                features = extract_features(str(outfile))
                features = features.reshape(1, -1)
                prediction = best_rf_model.predict(features)
                self.prediction = "Healthy" if prediction == 1 else "Not Healthy"
            except Exception as e:
                self.prediction = f"Error processing file: {str(e)}"

def index():
    return rx.vstack(
        rx.heading("Parkinson's Health Status Prediction"),
        rx.upload(
            rx.vstack(
                rx.button(
                    "Select Audio File",
                    bg="white",
                    border="1px solid rgb(107,99,246)",
                ),
                rx.text(
                    "Drag and drop audio files here or click to select files"
                ),
            ),
            id="audio_upload",
            accept={
                "audio/*": [".wav", ".mp3"]
            },
            multiple=False,
            border="1px dotted rgb(107,99,246)",
            padding="5em",
        ),
        rx.button(
            "Process Audio",
            on_click=State.handle_upload(
                rx.upload_files(upload_id="audio_upload")
            ),
        ),
        rx.heading(State.prediction),
        padding="5em",
    )

app = rx.App()
app.add_page(index)
