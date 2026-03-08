# Advanced Time Series Forecasting Engine

A comprehensive Python-based forecasting application built with Streamlit that supports multiple forecasting methods with automatic model comparison and validation.

## 🌟 Features

- **Multiple Forecasting Methods**:
  - SARIMA (Seasonal AutoRegressive Integrated Moving Average)
  - Holt-Winters (Exponential Smoothing)
  - TBATS (Trigonometric seasonality, Box-Cox, ARMA errors, Trend, Seasonal)
  - Prophet (Facebook's forecasting tool)
  - XGBoost (Gradient Boosting with engineered features)

- **Train-Test Validation**: Automatic split with customizable test size
- **Model Comparison**: Side-by-side comparison with automatic best model recommendation
- **Multiple Metrics**: RMSE, MAE, MAPE, MSE
- **Interactive Visualizations**: Using Plotly for dynamic charts
- **CSV Upload**: Easy data import functionality
- **Downloadable Results**: Export forecasts for each model

## 📋 Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies

## 🚀 Installation

### 1. Clone or Download

```bash
git clone <your-repo-url>
cd forecasting_app
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 💻 Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using the App

1. **Upload Data**:
   - Click "Browse files" in the sidebar
   - Upload a CSV file with time series data
   - Your CSV should have at least two columns: date and value

2. **Configure Settings**:
   - Select the date column from your data
   - Select the value column to forecast
   - Set forecast horizon (number of future periods)
   - Set test size (percentage for validation)

3. **Choose Models**:
   - Select which forecasting methods to use
   - Configure advanced parameters if needed

4. **Run Forecast**:
   - Click "Run Forecasting" button
   - Wait for models to train and validate
   - View results in multiple tabs

5. **Analyze Results**:
   - **Data Overview**: See your data and statistics
   - **Forecasts**: View individual model predictions
   - **Model Comparison**: Compare all models side-by-side
   - **Detailed Results**: Download forecasts and see detailed metrics

## 📊 Sample Data Format

Your CSV file should look like this:

```csv
date,value
2023-01-01,100
2023-01-02,105
2023-01-03,103
2023-01-04,108
...
```

## 🔧 Configuration Options

### Forecasting Parameters
- **Forecast Horizon**: 1-365 periods (default: 30)
- **Test Size**: 10-40% of data (default: 20%)

### Model-Specific Settings
- **SARIMA**: Customize order (p,d,q) and seasonal order (P,D,Q,s)
- **Prophet**: Choose additive or multiplicative seasonality

## 📈 Metrics Explained

- **RMSE (Root Mean Square Error)**: Square root of average squared differences
- **MAE (Mean Absolute Error)**: Average absolute differences
- **MAPE (Mean Absolute Percentage Error)**: Average percentage error
- **MSE (Mean Squared Error)**: Average squared differences

Lower values indicate better model performance.

## 🏆 Model Selection

The application automatically recommends the best model based on RMSE (Root Mean Square Error). The model with the lowest RMSE on the test set is highlighted as the recommended model.

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Date Parsing Issues**: Ensure your date column is in a standard format (YYYY-MM-DD, MM/DD/YYYY, etc.)

3. **Memory Issues**: For large datasets, reduce the test size or forecast horizon

4. **Model Failures**: Some models may fail for certain datasets. The app will skip failed models and continue with others.

## 📦 Deployment

### Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Deploy!

### Deploy to Heroku

1. Create a `Procfile`:
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. Deploy using Heroku CLI:
   ```bash
   heroku create
   git push heroku main
   ```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

Created with ❤️ for time series forecasting enthusiasts

## 🙏 Acknowledgments

- Streamlit for the amazing framework
- Facebook Prophet team
- Statsmodels contributors
- XGBoost developers
- TBATS package maintainers

## 📞 Support

For issues, questions, or suggestions, please open an issue on GitHub.

---

**Happy Forecasting! 📈**
