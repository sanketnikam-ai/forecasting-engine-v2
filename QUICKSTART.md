# Quick Start Guide

## Installation (5 minutes)

### Step 1: Prerequisites
- Python 3.8 or higher installed
- pip package manager

### Step 2: Setup
```bash
# Navigate to the project directory
cd forecasting_app

# Create virtual environment (recommended)
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
   - Click "🚀 Run Forecasting" button
   - Wait 1-2 minutes for results

5. **View Results**:
   - Check the "📉 Model Comparison" tab to see which model performed best
   - Explore "📈 Forecasts" tab for detailed predictions
   - Download forecasts from "📋 Detailed Results" tab

## Tips for Best Results

### Data Requirements
- **Minimum**: 50-100 data points recommended
- **Frequency**: Regular time intervals (daily, weekly, monthly)
- **Quality**: Minimal missing values
- **Pattern**: Clear trend or seasonal pattern helps

### Model Selection Guide

| Model | Best For | Data Type |
|-------|----------|-----------|
| **SARIMA** | Clear seasonal patterns | Stationary data with seasonality |
| **Holt-Winters** | Trend + Seasonality | Regular seasonal data |
| **TBATS** | Multiple seasonal patterns | Complex seasonality |
| **Prophet** | Daily data with holidays | Strong seasonal patterns |
| **XGBoost** | Complex patterns | Non-linear relationships |

### Configuration Tips
- **Start simple**: Use default settings first
- **Test size**: Start with 20%, increase for more validation
- **Forecast horizon**: Don't exceed 30% of your data length
- **Seasonality**: 
  - Monthly data → seasonal period = 12
  - Weekly data → seasonal period = 52
  - Daily data → seasonal period = 7 or 365

## Common Use Cases

### 1. Sales Forecasting
```
Date Column: order_date
Value Column: sales_amount
Horizon: 30 days
Models: All (compare performance)
```

### 2. Inventory Planning
```
Date Column: week
Value Column: stock_level
Horizon: 12 weeks
Models: Holt-Winters, Prophet
```

### 3. Demand Forecasting
```
Date Column: date
Value Column: demand
Horizon: 90 days
Models: Prophet, XGBoost, SARIMA
```

### 4. Traffic Prediction
```
Date Column: timestamp
Value Column: visitors
Horizon: 7 days
Models: Prophet, TBATS
```

## Troubleshooting

### "Model failed" warning
✅ **Normal behavior** - some models may not fit all datasets
- Other models will continue running
- At least 2-3 models should succeed
- Check if your data has enough points

### Slow performance
⚡ **Solutions**:
- Reduce forecast horizon
- Use fewer models (start with 2-3)
- Try smaller dataset for testing
- Increase test size to reduce training data

### Import errors
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### Prophet-specific errors
```bash
# Install Prophet dependencies first
pip install pystan==2.19.1.1
pip install prophet
```

### Date parsing issues
- Ensure dates are in standard format: `YYYY-MM-DD` or `MM/DD/YYYY`
- Check for missing dates in your sequence
- Verify date column doesn't have text values

## Understanding Results

### Metrics Interpretation

| Metric | What it means | Good value |
|--------|--------------|------------|
| **RMSE** | Average prediction error | Lower is better |
| **MAE** | Average absolute error | Lower is better |
| **MAPE** | Percentage error | < 10% is excellent |
| **MSE** | Squared errors | Lower is better |

### Model Selection
- 🏆 **Best Model**: Shown in green in comparison tab
- Based on lowest RMSE on test data
- Consider business context too!

### Forecast Interpretation
- **Blue line**: Historical training data
- **Green line**: Actual test data
- **Orange dashed**: Model predictions on test
- **Red dotted**: Future forecast

## Next Steps

1. ✅ **Verify Results**: Check if predictions make business sense
2. 📊 **Export Data**: Download forecasts for reporting
3. 🔄 **Iterate**: Try different parameters
4. 📈 **Monitor**: Re-run with new data regularly
5. 🎯 **Productionize**: Deploy to Streamlit Cloud

## Example Workflow

```
1. Upload CSV ✅
2. Select date & value columns ✅
3. Start with default settings ✅
4. Run all 5 models ✅
5. Check Model Comparison tab ✅
6. Note best performing model ✅
7. Download that model's forecast ✅
8. Use in your business planning ✅
```

## Pro Tips

💡 **Data Quality Matters**
- Clean outliers before forecasting
- Fill missing values appropriately
- Ensure consistent time intervals

💡 **Model Ensemble**
- Consider averaging top 2-3 models
- Can improve robustness
- Reduces single-model risk

💡 **Seasonal Adjustments**
- Enable seasonal parameters for periodic data
- Use Prophet for complex seasonality
- SARIMA for classical seasonal patterns

💡 **Regular Updates**
- Retrain models monthly with new data
- Monitor forecast accuracy
- Adjust parameters based on performance

## Resources

- 📖 Full documentation: `README.md`
- 💻 Source code: `app.py`
- 📊 Sample data: `sample_data.csv`
- 🔧 Advanced settings: Available in sidebar expander

## Getting Help

If you encounter issues:
1. Check this guide first
2. Review `README.md` for detailed info
3. Verify your data format
4. Try with `sample_data.csv` to isolate issues
5. Check error messages in the app

---

**Ready to forecast? Upload your data and let's go! 🚀**
