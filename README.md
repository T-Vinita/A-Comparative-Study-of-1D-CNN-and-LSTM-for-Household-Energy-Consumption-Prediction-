````markdown name=README.md url=https://github.com/T-Vinita/A-Comparative-Study-of-1D-CNN-and-LSTM-for-Household-Energy-Consumption-Prediction-/blob/main/README.md
# Electricity Consumption Forecasting using LSTM and 1D-CNN

A deep learning project that predicts hourly electricity consumption and compares LSTM and 1D-CNN models to evaluate accuracy and stability. This repository contains code, notebooks, and results used to preprocess the dataset, train models, evaluate performance, and visualize outcomes.

---

## Overview

This project focuses on time-series forecasting of hourly household electricity consumption. We implement two deep learning approaches — Long Short-Term Memory (LSTM) and 1D Convolutional Neural Network (1D-CNN) — and compare them using standard regression metrics. The goal is to find which model gives better accuracy and produces more stable predictions for short-term forecasting.

---

## Problem Statement

Forecast hourly electricity consumption for a household (or aggregated meter) using historical consumption data. Accurate short-term forecasts help with grid balancing, energy planning, and smart-home energy management.

Key questions:
- Which model, LSTM or 1D-CNN, provides better forecasting accuracy for hourly consumption?
- Which model is more stable (consistent) across different validation splits?

---

## Dataset

- Type: Time-series — hourly electricity consumption records.
- Typical columns: `timestamp` (datetime), `consumption` (kWh or similar).
- Preprocessing performed:
  - Missing value handling and interpolation
  - Aggregation (if needed)
  - Time-based feature extraction (hour of day, day of week, holiday flags — optional)
  - Normalization/standardization before training
- Note: This README assumes you have a CSV or similar file with hourly consumption data. Replace the dataset path in scripts/notebooks as needed.

---

## Models Used

- LSTM (Long Short-Term Memory)
  - RNN variant that captures long-term dependencies in sequences.
  - Often effective for time-series with temporal patterns such as daily/weekly seasonality.

- 1D Convolutional Neural Network (1D-CNN)
  - Uses temporal convolutions to learn local patterns in the sequence.
  - Can be faster to train and capture local trends and short-term dependencies.

Both models are implemented using TensorFlow / Keras and trained on sequences generated from the normalized time-series.

---

## Tech Stack

- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Matplotlib (and optionally Seaborn)
- Jupyter Notebook (for exploratory analysis and experiments)

---

## Project Workflow

1. Data loading:
   - Load CSV / time-series file and set datetime index.
2. Data cleaning & EDA:
   - Inspect missing values, outliers, and seasonality.
   - Visualize raw series (time plot, histograms).
3. Feature engineering:
   - Extract time features (hour, weekday).
   - Optionally include lag features and rolling statistics.
4. Normalization:
   - Scale features (MinMax or StandardScaler) using training set statistics.
5. Sequence generation:
   - Convert series into supervised learning sequences (X sequences, y targets).
   - Choose look-back window (e.g., 24 or 168 hours).
6. Train / Validation / Test split:
   - Use time-aware splitting (no random shuffle).
7. Model training:
   - Train LSTM and 1D-CNN separately with early stopping, checkpoints.
8. Evaluation:
   - Compute MAE, MSE, RMSE on validation/test sets.
   - Compare accuracy and stability across runs.
9. Visualization & reporting:
   - Plot predictions vs actual, loss curves, error distributions.
10. Conclusions and next steps.

---

## Evaluation Metrics

We use common regression metrics that are easy to interpret:

- Mean Absolute Error (MAE)
  - MAE = mean(|y_true - y_pred|)
  - Shows average absolute error in the same units as the target.

- Mean Squared Error (MSE)
  - MSE = mean((y_true - y_pred)^2)
  - Penalizes larger errors more heavily.

- Root Mean Squared Error (RMSE)
  - RMSE = sqrt(MSE)
  - Same units as the target and more sensitive to large errors.

These metrics help compare model accuracy and quantify stability (e.g., standard deviation of metric over multiple runs/splits).

---

## Results & Observations

- Key result: LSTM outperformed 1D-CNN in both accuracy and stability for this dataset.
  - LSTM typically achieved lower MAE, MSE, and RMSE on the held-out test set.
  - LSTM predictions were more consistent across multiple validation splits and training runs.
- 1D-CNN:
  - Faster to train and sometimes better at capturing very short-term local patterns.
  - Slightly higher variance and larger errors for longer horizons in this experiment.

Note: Exact numbers depend on dataset, look-back window, hyperparameters, and preprocessing. See the notebooks / results folder for detailed metric tables and experiment logs.

---

## Visualizations

Important plots produced during the project:

- Time-series plot of raw consumption (full period).
- Train / Validation loss curves for each model.
- Predictions vs Actual (line plots) for test period.
- Scatter plot of predicted vs actual values (parity plot).
- Error distribution histograms (residuals).
- Rolling-window metrics (to show stability over time).

These visualizations make it easy to see where models succeed or fail (e.g., peaks, seasonality, sudden changes).

---

## Key Learnings

- LSTM is well suited for hourly electricity forecasting where temporal dependencies span many time steps (daily/weekly patterns).
- 1D-CNN is a strong baseline: it is faster, simpler, and captures local patterns — but may struggle with longer-term dependencies unless designed carefully.
- Proper preprocessing (handling missing data, scaling, appropriate sequence generation) has a large impact on final performance.
- Use time-aware splits (no random shuffling) to avoid data leakage in time-series problems.
- Early stopping and model checkpoints are helpful to prevent overfitting.
- Run multiple experiments (different seeds / splits) to measure model stability, not just a single metric.

---

## Future Enhancements

- Hyperparameter search (Grid / Bayesian) for architecture and training settings.
- Add exogenous features (temperature, occupancy, holidays) to improve predictions.
- Use hybrid models (CNN + LSTM) or attention mechanisms (Transformer-based models).
- Multi-step forecasting or probabilistic forecasting (quantile loss).
- Deploy a simple API for real-time inference and monitoring.
- Cross-validation adapted for time-series (walk-forward validation) to better estimate performance.

---

## How to run (quick start)

1. Clone the repository:
   - git clone https://github.com/T-Vinita/A-Comparative-Study-of-1D-CNN-and-LSTM-for-Household-Energy-Consumption-Prediction-.git

2. Install dependencies:
   - pip install -r requirements.txt
   (or create a conda environment and install listed packages)

3. Prepare data:
   - Place your hourly CSV (columns: timestamp, consumption) in the `data/` folder or update paths in the notebook/scripts.

4. Run notebooks or scripts:
   - Open the main notebook (e.g., `notebooks/experiment.ipynb`) and run cells in order, or run training scripts:
     - python scripts/train_lstm.py
     - python scripts/train_cnn1d.py

5. Results and plots will be saved in `results/` or displayed in the notebook.

---

## Contact

Author: T-Vinita  
Repository: https://github.com/T-Vinita/A-Comparative-Study-of-1D-CNN-and-LSTM-for-Household-Energy-Consumption-Prediction-

If you have questions or want to discuss improvements, feel free to open an issue or submit a PR.

---
````
