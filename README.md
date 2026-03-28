# Advanced Time Series Forecasting Engine

A comprehensive Python forecasting application with **22+ algorithms** (25+ with optional models). Built with Streamlit. Optimized for monthly YYYYMM format.

## 🌟 Features

### 22+ Forecasting Algorithms (Core Installation)

**📊 Statistical Models (12):**
- Naive Forecast
- Seasonal Naive  
- Simple Moving Average (SMA)
- Weighted Moving Average (WMA)
- Exponential Smoothing (SES)
- Holt-Winters
- ARIMA
- SARIMA
- Auto ARIMA
- ARIMAX
- TBATS
- Croston's Method

**🤖 Machine Learning (7):**
- Linear Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- Gradient Boosting (GBM)
- Ensemble (averages all models)

**🧠 Deep Learning (3):**
- Facebook Prophet (Core)
- Neural Prophet (Optional*)
- LSTM (Optional*)

*Install separately via `requirements-optional.txt`

### Key Features:
- ✅ **YYYYMM Format** - Native support (202301, 202302...)
- ✅ **Train-Test Validation** - Customizable split
- ✅ **Auto Best Model** - RMSE-based recommendation
- ✅ **Multiple Metrics** - RMSE, MAE, MAPE
- ✅ **Interactive Charts** - Plotly visualizations
- ✅ **CSV Import/Export** - Easy data handling
- ✅ **Organized UI** - Models grouped by category

## 📋 Requirements

- Python 3.8+
- Core dependencies in `requirements.txt`

### Installation Options:

**Option A: Core (Recommended - 22 models)**
```bash
pip install -r requirements.txt
```

**Option B: Full (Optional +2 models)**
```bash
pip install -r requirements.txt
pip install -r requirements-optional.txt  # Adds Neural Prophet, LSTM
```

## 🚀 Quick Start

```bash
# Clone/download
cd forecasting_app

# Install
pip install -r requirements.txt

# Run
streamlit run app.py
```

Open browser at `http://localhost:8501`

## 💻 Usage

1. **Upload CSV**: YYYYMM format
2. **Select Columns**: Date and value
3. **Choose Models**: From 22+ algorithms
4. **Configure**: Horizon and test size
5. **Run**: Click button
6. **Analyze**: Compare and download

## 📊 Data Format

### YYYYMM (Recommended):
```csv
month,sales
202301,12450
202302,13200
202303,14100
```

**Also supports:** YYYY-MM-DD, YYYY-MM, MM/YYYY

## 🎯 Model Selection Guide

| Use Case | Recommended |
|----------|-------------|
| Simple trend | Naive, SMA, Linear |
| Seasonal data | SARIMA, Holt-Winters, TBATS |
| Complex patterns | XGBoost, Prophet, Ensemble |
| Best accuracy | Auto ARIMA, Ensemble |
| Fast results | Naive, SMA, Linear |

## 📈 Metrics

- **RMSE**: Root Mean Squared Error (lower = better)
- **MAE**: Mean Absolute Error (lower = better)
- **MAPE**: Mean Absolute Percentage Error (lower = better)

## 🔧 Configuration

- **Horizon**: 1-60 periods (default: 12)
- **Test Size**: 10-40% (default: 20%)

## 🐛 Troubleshooting

### Installation Issues:

```bash
# Prophet fails
pip install pystan==2.19.1.1
pip install prophet

# Optional models (not required)
pip install -r requirements-optional.txt
```

### Model Failures:
- Normal - some models may fail on certain datasets
- App skips failed models and continues
- Basic models (Naive, SMA) always work

## 📦 Deployment

### Streamlit Cloud:
Use only `requirements.txt` (not optional)

### Heroku:
```bash
heroku create
git push heroku main
```

## 📂 Project Structure

```
forecasting_app/
├── app.py                      # Main app (22+ models)
├── requirements.txt            # Core dependencies
├── requirements-optional.txt   # Optional (Neural Prophet, LSTM)
├── packages.txt               # System dependencies
├── sample_data.csv            # Sample YYYYMM data
├── README.md                  # This file
├── QUICKSTART.md              # 5-min guide
├── INSTALLATION.md            # Install guide
├── .gitignore
├── LICENSE
└── .streamlit/config.toml
```

## 🤝 Contributing

Contributions welcome! Open an issue or PR.

## 📄 License

MIT License

## 🙏 Acknowledgments

Streamlit, Prophet, Statsmodels, XGBoost, LightGBM, CatBoost, pmdarima, TBATS

## 📊 Version

**v2.0** - 22+ Models with YYYYMM Support

---

**Happy Forecasting! 📈**
