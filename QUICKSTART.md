# Quick Start Guide

## 🚀 5-Minute Setup

### 1. Install
```bash
cd forecasting_app
pip install -r requirements.txt
```

### 2. Run
```bash
streamlit run app.py
```

### 3. Upload Data
- Use `sample_data.csv` or your own
- Format: YYYYMM (202301, 202302...)

### 4. Configure
- Select date & value columns
- Set forecast horizon (12 months)
- Choose test size (20%)

### 5. Select Models
- Statistical: Naive, ARIMA, SARIMA, Holt-Winters
- ML: Random Forest, XGBoost, LightGBM
- DL: Prophet
- Ensemble: Combines all

### 6. Run & Analyze
- Click "Run Forecasting"
- View results in tabs
- Download forecasts

## 📊 Model Selection Guide

| Use Case | Recommended Models |
|----------|-------------------|
| Simple trend | Naive, SMA, Linear |
| Seasonal data | SARIMA, Holt-Winters, TBATS |
| Complex patterns | XGBoost, Prophet, Ensemble |
| Fast results | Naive, SMA, WMA |
| Best accuracy | Auto ARIMA, Ensemble, Prophet |

## 🎯 Tips

- Start with 3-5 models for speed
- Use Ensemble for robustness
- Monthly data needs 24+ months
- YYYYMM format is simplest

## 📈 Example Workflow

```
Upload → Select Columns → Choose 5 Models → Run → Compare → Download Best
```

**Done! Start forecasting!** 🚀
