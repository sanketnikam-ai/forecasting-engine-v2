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

### 3. Upload
- Use `sample_data.csv`
- Format: YYYYMM (202301, 202302...)

### 4. Configure
- Date & value columns
- Horizon: 12 months
- Test: 20%

### 5. Select Models
Pick 3-5 to start:
- Statistical: SARIMA, Holt-Winters
- ML: XGBoost, Random Forest
- DL: Prophet
- Ensemble

### 6. Run & Download
- Click "Run Forecasting"
- Compare results
- Download best model

## 📊 Model Guide

| Use Case | Models |
|----------|--------|
| Simple | Naive, SMA, Linear |
| Seasonal | SARIMA, Holt-Winters |
| Complex | XGBoost, Prophet, Ensemble |
| Fast | Naive, SMA |
| Accurate | Auto ARIMA, Ensemble |

## 🎯 Tips

- Start with 3-5 models
- Use Ensemble for robustness
- Need 24+ months data
- YYYYMM is simplest

## Workflow

```
Upload → Columns → Models → Run → Compare → Download
```

**Done! 🚀**
