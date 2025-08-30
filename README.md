## Breast Cancer Classification

A machine learning project for classifying breast cancer (benign vs malignant) using classical ML models and an ensemble. The repository includes the data, a full exploratory notebook, and exported artifacts (trained ensemble model and feature scaler).

### Features
- **End-to-end notebook**: data loading, EDA, preprocessing, modeling, evaluation
- **Saved artifacts**: `best_model_ensemble.joblib`, `scaler.joblib`/`scaler.pkl`
- **Dataset**: `breast_cancer.csv` (tabular features)

### Project structure
```
Breast_Cancer_Classification/
├─ breast_Cancer.ipynb           # Main notebook with the full workflow
├─ breast_cancer.csv             # Dataset
├─ best_model_ensemble.joblib    # Trained ensemble model
├─ scaler.joblib                 # Trained feature scaler (Joblib format)
├─ scaler.pkl                    # Trained feature scaler (Pickle format)
├─ LICENSE                       # License for this project
└─ README.md                     # You are here
```

### Environment setup
Requirements (tested with Python 3.9+):
- numpy
- pandas
- scikit-learn
- joblib
- matplotlib
- seaborn
- jupyter

Create and activate a virtual environment, then install dependencies:
```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
pip install --upgrade pip
pip install numpy pandas scikit-learn joblib matplotlib seaborn jupyter
```

### Run the notebook
```bash
jupyter notebook breast_Cancer.ipynb
```
Run all cells to reproduce preprocessing, training, evaluation, and export of artifacts.

### Use the saved model for inference
Below is a minimal example to load the scaler and ensemble model to make a prediction. Adapt the feature names/order to match your dataset columns.

```python
import joblib
import numpy as np
import pandas as pd

# Load artifacts
model = joblib.load("best_model_ensemble.joblib")
try:
    scaler = joblib.load("scaler.joblib")
except Exception:
    # Fallback if scaler was saved as pickle
    import pickle
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

# Example input (replace with your real feature values)
sample = {
    # "mean_radius": 14.1,
    # "mean_texture": 20.0,
    # ... add all required features in the correct order ...
}

# Convert to DataFrame and scale
X = pd.DataFrame([sample])
X_scaled = scaler.transform(X)

# Predict
pred = model.predict(X_scaled)[0]
proba = getattr(model, "predict_proba", lambda X: None)(X_scaled)

print("Predicted class:", pred)
if proba is not None:
    print("Class probabilities:", proba[0])
```

Tips:
- Ensure the input `sample` includes all features used during training and in the same order/column names.
- If your scaler expects a fixed column order, build the `DataFrame` with those exact columns.

### Reproducibility
Results may vary slightly due to randomness in model training. For consistent runs, set random seeds in the notebook where applicable (e.g., `random_state` in scikit-learn models and splits).

### License
This project is licensed — see `LICENSE` for details.

### Contributing
Issues and pull requests are welcome. Please describe proposed changes clearly and keep the notebook output clean where possible.


