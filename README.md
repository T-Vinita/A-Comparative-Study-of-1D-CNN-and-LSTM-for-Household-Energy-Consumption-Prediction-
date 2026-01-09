# ⚡ Electricity Consumption Forecasting using LSTM and 1D-CNN

Welcome to the Electricity Consumption Forecasting project!  
This repository uses deep learning to predict hourly electricity consumption and compares LSTM and 1D-CNN models to find which one gives better accuracy and stability. Ideal for learning time-series forecasting, preparing for interviews, or building a simple forecasting pipeline. 🚀

---

## 🕵️ About the Project

Hourly electricity usage often follows daily and weekly patterns. We build and compare two neural architectures:

- LSTM (Long Short-Term Memory) — good at learning long-term temporal patterns.
- 1D-CNN (1D Convolutional Neural Network) — good at learning local temporal features quickly.

Goal: Predict the next hours of consumption and compare models on accuracy and stability.

---

## 🧩 How It Works

### 1️⃣ Data Preparation
- Load hourly consumption data and set a datetime index.
- Handle missing values (interpolation / forward-fill).
- Extract time features (hour of day, day of week) if needed.
- Normalize/scale data using statistics from the training set.

### 2️⃣ Sequence Generation
- Convert time-series into supervised sequences (X windows → y next-step or multi-step).
- Choose look-back window (e.g., last 24 or 168 hours).

### 3️⃣ Model Training
- Train LSTM and 1D-CNN models with early stopping and checkpoints.
- Use time-aware splits (train → validation → test) — no shuffling.

### 4️⃣ Evaluation & Comparison
- Evaluate with MAE, MSE, RMSE.
- Compare models for accuracy and consistency across multiple runs/splits.

---

## ����️ Interactive Demo (Quick Start)

🎮 Try it yourself:

1. Clone this repo:
   ```bash
   git clone https://github.com/T-Vinita/A-Comparative-Study-of-1D-CNN-and-LSTM-for-Household-Energy-Consumption-Prediction-.git
   cd A-Comparative-Study-of-1D-CNN-and-LSTM-for-Household-Energy-Consumption-Prediction-
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare your data:
   - Place your hourly CSV (columns: timestamp, consumption) in the `data/` folder or update script paths.

4. Run the notebook or training scripts:
   - Jupyter:
     ```bash
     jupyter notebook
     ```
     Open `notebooks/experiment.ipynb`.
   - or run scripts:
     ```bash
     python scripts/train_lstm.py
     python scripts/train_cnn1d.py
     ```

5. View results and plots in `results/` or directly in the notebook.

---

## 🖼️ Visualize the Results

See model performance with:

- Predictions vs Actual (time series)
- Train / Validation loss curves
- Prediction parity (scatter) plots
- Residual / error histograms
- Rolling-window metrics to show stability

<p align="center">
  <img src="docs/images/forecast_sample.png" alt="Forecast Visualization" width="600"/>
</p>

---

## 💡 Key Features

- Time-series specific pipeline: cleaning → scaling → sequence generation.
- Two model comparisons: LSTM vs 1D-CNN.
- Time-aware splitting (no data leakage).
- Early stopping and model checkpoints to avoid overfitting.
- Simple notebooks and scripts for reproducible experiments.

---

## 📝 Dataset

- Type: Hourly electricity consumption.
- Typical columns: `timestamp`, `consumption` (kWh).
- Preprocessing used:
  - Missing value handling (interpolation / forward fill)
  - Optional time features (hour, weekday)
  - Scaling using MinMax or StandardScaler (fit on training set)
- Source: Place your own CSV or use publicly available household/utility datasets.

---

## 🧠 Models Used

- LSTM (Long Short-Term Memory)
  - Captures temporal dependencies across longer horizons.
  - Good at learning daily/weekly seasonality.

- 1D-CNN (1D Convolutional Neural Network)
  - Learns local temporal patterns quickly.
  - Faster training, effective for short-term dependencies.

Both are implemented with TensorFlow / Keras and trained on sequences generated from the normalized series.

---

## 🧰 Tech Stack

- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Matplotlib (and optionally Seaborn)
- Jupyter Notebook

---

## 📈 Evaluation Metrics

| Metric | Description |
| ------ | ----------- |
| MAE (Mean Absolute Error) | Average absolute difference between predictions and actuals |
| MSE (Mean Squared Error) | Average squared error — penalizes large errors |
| RMSE (Root MSE) | Square root of MSE — same units as target |

Use these metrics to compare accuracy and to measure stability across runs.

---

## 📊 Results & Observations

- Key result: LSTM outperformed 1D-CNN in accuracy and stability on this dataset.
  - LSTM produced lower MAE, MSE, and RMSE on the test set.
  - LSTM predictions were more consistent across different validation splits.
- 1D-CNN:
  - Trained faster and captured short-term patterns well.
  - Slightly higher error for longer-horizon predictions in this experiment.
- Exact numbers depend on look-back window, preprocessing choices, and hyperparameters — check the notebooks for experiment logs and tables.

---

## 🔎 Key Learnings

- Proper preprocessing (missing values, scaling, correct sequence generation) is crucial.
- Time-aware splitting prevents data leakage — always use it for time-series.
- LSTM is effective for series with longer temporal dependencies (daily/weekly cycles).
- 1D-CNN is a good, fast baseline for short-term patterns.
- Evaluate stability (run multiple seeds / walk-forward validation) not just single-run metrics.

---

## 🚀 Future Enhancements

- Hyperparameter tuning (Grid / Bayesian search).
- Add exogenous features (temperature, occupancy, holidays).
- Try hybrid models (CNN + LSTM) or attention-based models (Transformers).
- Multi-step forecasting and probabilistic forecasting (prediction intervals, quantiles).
- Implement walk-forward cross-validation for more robust evaluation.
- Build a simple API for real-time inference and monitoring.

---

## 🤝 Contributing

Want to help improve the project?  
- Open an issue to discuss ideas.
- Submit a pull request with improvements (code, docs, tests).
- Share additional datasets or benchmarking scripts.

---

## 📢 Contact

Author: T-Vinita  
Repo: https://github.com/T-Vinita/A-Comparative-Study-of-1D-CNN-and-LSTM-for-Household-Energy-Consumption-Prediction-

If you have questions or suggestions, open an issue or start a discussion — happy to help!

---

> Can LSTM beat 1D-CNN every time? Not always — try different datasets and settings and report what you find. Happy forecasting! ⚡
