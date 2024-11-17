# ![SpeechReclaim Logo](assets/teamlogo.png) SpeechReclaim: Parkinson's Disease Speech Diagnostics

This repository contains the code for **SpeechReclaim**, a machine learning project aimed at developing a diagnostic tool for Parkinson's Disease using speech analysis. The model utilizes voice features extracted from audio recordings to classify individuals as healthy or potentially affected by Parkinson's.

## Project Overview

Parkinson's Disease (PD) can impact speech patterns. Early detection of PD is crucial for managing the disease and improving the quality of life of patients. This project uses machine learning to analyze speech recordings and identify features related to Parkinson's Disease.

### Features:
- Audio feature extraction from speech samples
- Preprocessing and cleaning of Parkinson's speech data
- Model training with Random Forest classifier
- Hyperparameter tuning for optimal performance
- Evaluation using accuracy, precision, recall, F1-score, and ROC curve

## Requirements

- Python 3.6+
- Pandas
- NumPy
- scikit-learn
- librosa
- Reflex (for web deployment)
- Joblib
- Matplotlib

Use the following command to run the app:

```bash
reflex init 
reflex run

