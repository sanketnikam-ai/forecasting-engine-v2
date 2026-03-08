# Quick Start Guide

## Installation (5 minutes)

### Step 1: Prerequisites
- Python 3.8 or higher installed
- pip package manager

### Step 2: Setup
```bash
# Navigate to the project directory
cd forecasting_app

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Run the Application
```bash
streamlit run app.py
```

The app will open automatically in your browser at http://localhost:8501

## First Forecast (2 minutes)

1. **Test with Sample Data**:
   - Use the included `sample_data.csv` file
   - Or upload your own CSV with date and value columns

2. **Quick Configuration**:
   - Select "date" as Date Column
   - Select "sales" (or your value column) as Value Column
   - Keep default settings (30 days forecast, 20% test size)

3. **Select Models**:
   - Check all boxes (SARIMA, Holt-Winters, TBATS, Prophet, XGBoost)

4. **Run**:
   - Click "Run Forecasting" button
   - Wait 1-2 minutes for results

5. **View Results**:
   - Check the "Model Comparison" tab to see which model performed best
   - Explore "Forecasts" tab for detailed predictions
   - Download forecasts from "Detailed Results" tab

## Tips for Best Results

### Data Requirements
- At least 50-100 data points recommended
- Regular time intervals (daily, weekly, monthly)
- Minimal missing values
- Clear trend or seasonal pattern

### Model Selection
- **SARIMA**: Best for data with clear seasonality
- **Holt-Winters**: Good for trend + seasonality
- **TBATS**: Handles multiple seasonal patterns
- **Prophet**: Robust to missing data and outliers
- **XGBoost**: Best for complex non-linear patterns

### Configuration Tips
- Start with 20% test size
- Increase forecast horizon gradually
- For monthly data, set seasonal period to 12
- For weekly data, set seasonal period to 52

## Troubleshooting

### "Model failed" warning
- Normal - some models may not fit all datasets
- Other models will continue running
- At least 2-3 models should succeed

### Slow performance
- Reduce forecast horizon
- Use fewer models
- Try smaller dataset for testing

### Import errors
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

## Next Steps

1. Try with your own data
2. Experiment with different parameters
3. Compare model performances
4. Export forecasts for your reports

## Support

- Check README.md for detailed documentation
- Review app.py for code details
- Refer to model-specific documentation for advanced tuning

---

Happy Forecasting! 🚀
