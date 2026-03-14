# Advanced Time Series Forecasting Engine

A comprehensive Python-based forecasting application with **25+ forecasting algorithms** built with Streamlit. Optimized for monthly business data in YYYYMM format.

## 🌟 Features

### 25+ Forecasting Algorithms

**📊 Statistical Models (12):**
- Naive Forecast
- Seasonal Naive
- Simple Moving Average (SMA)
- Weighted Moving Average (WMA)
- Exponential Smoothing (SES)
- Holt-Winters (Triple Exponential Smoothing)
- ARIMA
- SARIMA
- Auto ARIMA (automatic parameter selection)
- ARIMAX (ARIMA with exogenous variables)
- TBATS (Complex seasonal patterns)
- Croston's Method (Intermittent demand)

**🤖 Machine Learning Models (7):**
- Linear Regression
- Multilinear Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- Gradient Boosting Machines (GBM)

**🧠 Deep Learning Models (3):**
- Facebook Prophet
- Neural Prophet
- LSTM (Long Short-Term Memory)

**🎯 Ensemble:**
- Ensemble Model (averages all successful models)

### Additional Features:
- **YYYYMM Format Support**: Native YYYYMM (202301, 202302) parsing
- **Train-Test Validation**: Customizable test split
- **Auto Model Selection**: Best model recommendation based on RMSE
- **Multiple Metrics**: RMSE, MAE, MAPE, MSE
- **Interactive Visualizations**: Plotly charts
- **CSV Import/Export**: Easy data handling
- **Organized UI**: Models grouped by category

## 📋 Requirements

- Python 3.8+
- See `requirements.txt` for dependencies

## 🚀 Quick Start

### Installation

```bash
git clone <repository-url>
cd forecasting_app

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

Open browser at `http://localhost:8501`

## 💻 Usage

1. **Upload CSV**: YYYYMM format (202301, 202302, ...)
2. **Select Columns**: Date and value columns
3. **Choose Models**: From 25+ algorithms
4. **Configure**: Forecast horizon and test size
5. **Run**: Click "Run Forecasting"
6. **Analyze**: View results, compare models, download forecasts

## 📊 Data Format

### YYYYMM Format (Recommended):
```csv
month,sales
202301,12450
202302,13200
202303,14100
```

### Also Supports:
- YYYY-MM-DD: 2023-01-01
- YYYY-MM: 2023-01
- MM/YYYY: 01/2023

## 🎯 Model Categories

### When to Use Each Category:

**Statistical Models** - Best for:
- Clear seasonal patterns
- Traditional time series analysis
- Interpretability needed
- Smaller datasets

**Machine Learning** - Best for:
- Non-linear patterns
- Multiple features/variables
- Complex relationships
- Larger datasets

**Deep Learning** - Best for:
- Very large datasets
- Complex seasonal patterns
- Long-term dependencies
- Trend changes

**Ensemble** - Best for:
- Combining multiple approaches
- Reducing model risk
- Improving robustness

## 📈 Metrics Explained

- **RMSE**: Root Mean Squared Error (lower is better)
- **MAE**: Mean Absolute Error (lower is better)
- **MAPE**: Mean Absolute Percentage Error (lower is better)
- **MSE**: Mean Squared Error (lower is better)

## 🔧 Configuration

### Forecast Parameters:
- **Horizon**: 1-60 periods (default: 12 months)
- **Test Size**: 10-40% (default: 20%)

### Advanced Options:
- Individual model selection
- Category-based selection
- Ensemble configuration

## 🐛 Troubleshooting

### Installation Issues:
```bash
# If Prophet fails
pip install pystan==2.19.1.1
pip install prophet

# If TensorFlow/LSTM fails
pip install tensorflow==2.13.0
```

### Model Failures:
- Some models may fail on certain datasets
- App will skip failed models and continue
- At least basic models (Naive, SMA) should work

## 📦 Deployment

### Streamlit Cloud:
1. Push to GitHub
2. Connect at share.streamlit.io
3. Deploy

### Heroku:
```bash
heroku create
git push heroku main
```

## 📂 Project Structure

```
forecasting_app/
├── app.py                    # Main application (25+ models)
├── requirements.txt          # Python dependencies
├── packages.txt              # System dependencies
├── sample_data.csv           # Sample YYYYMM data
├── README.md                 # Documentation
├── QUICKSTART.md             # Quick guide
├── DATA_FORMAT_GUIDE.md      # Format conversion
├── .gitignore               # Git ignore
├── LICENSE                   # MIT License
├── CONTRIBUTING.md           # Contribution guide
└── .streamlit/
    └── config.toml           # Streamlit config
```

## 🤝 Contributing

Contributions welcome! See CONTRIBUTING.md

## 📄 License

MIT License - see LICENSE file

## 👨‍💻 Support

- Issues: GitHub Issues
- Documentation: README.md, QUICKSTART.md
- Format Guide: DATA_FORMAT_GUIDE.md

## 🙏 Acknowledgments

- Streamlit
- Facebook Prophet team
- Statsmodels contributors
- XGBoost, LightGBM, CatBoost developers
- TensorFlow team
- pmdarima (Auto ARIMA)
- TBATS package

## 🔄 Version

**Version 2.0.0** - 25+ Algorithms Edition

### Changelog:
- ✅ 25+ forecasting models
- ✅ YYYYMM format support
- ✅ Organized UI (Statistical/ML/DL categories)
- ✅ Auto ARIMA
- ✅ Ensemble methods
- ✅ Comprehensive error handling

---

**Happy Forecasting! 📈**
