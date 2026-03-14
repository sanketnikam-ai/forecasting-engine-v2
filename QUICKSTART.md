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
   - Use the included `sample_data.csv` file (50 months of data in YYYYMM format)
   - Or upload your own CSV with month (YYYYMM) and value columns

2. **Quick Configuration**:
   - Select "month" as Date Column
   - Select "sales" (or your value column) as Value Column
   - Keep default settings (12 months forecast, 20% test size)

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
- **Minimum**: 24-36 months of data recommended (2-3 years)
- **Format**: YYYYMM (202301, 202302, etc.) - simplest and recommended
- **Quality**: Minimal missing values, no gaps in months
- **Pattern**: Clear trend or seasonal pattern helps

### Model Selection Guide

| Model | Best For | Data Type |
|-------|----------|-----------|
| **SARIMA** | Clear seasonal patterns | Monthly data with seasonality |
| **Holt-Winters** | Trend + Seasonality | Regular monthly business data |
| **TBATS** | Multiple seasonal patterns | Complex monthly patterns |
| **Prophet** | Monthly data with trends | Strong seasonal patterns |
| **XGBoost** | Complex patterns | Non-linear monthly relationships |

### Configuration Tips
- **Start simple**: Use default settings first
- **Test size**: Start with 20%, increase for more validation
- **Forecast horizon**: 
  - 6 months: Short-term planning
  - 12 months: Annual planning
  - 24 months: Long-term strategy
- **Seasonality**: 
  - Monthly data → seasonal period = 12 (annual seasonality)
  - Quarterly data → seasonal period = 4

## Common Use Cases

### 1. Monthly Sales Forecasting
```
Date Column: month (YYYYMM: 202301, 202302...)
Value Column: sales_amount
Horizon: 12 months
Models: All (compare performance)
```

### 2. Inventory Planning
```
Date Column: month (YYYYMM)
Value Column: inventory_level
Horizon: 6 months
Models: Holt-Winters, Prophet
```

### 3. Revenue Forecasting
```
Date Column: month (YYYYMM)
Value Column: revenue
Horizon: 12 months
Models: Prophet, XGBoost, SARIMA
```

### 4. Demand Forecasting
```
Date Column: month (YYYYMM)
Value Column: units_sold
Horizon: 18 months
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
- **YYYYMM format** (recommended): 202301, 202302, 202303, etc.
- Ensure dates are consistent (all in same format)
- For monthly data, YYYYMM is the simplest format
- Alternative formats: `YYYY-MM-01`, `YYYY-MM`, or `MM/YYYY`
- Check for missing months in your sequence
- Verify date column doesn't have text values

**Example of correct YYYYMM data:**
```csv
month,sales
202201,10000
202202,12000
202203,11500
```

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
1. Upload CSV with YYYYMM format ✅
2. Select month & value columns ✅
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
- Use YYYYMM format for simplicity

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
- 📅 Date format guide: `DATA_FORMAT_GUIDE.md`
- 💻 Source code: `app.py`
- 📊 Sample data: `sample_data.csv` (YYYYMM format)
- 🔧 Advanced settings: Available in sidebar expander

## Getting Help

If you encounter issues:
1. Check this guide first
2. Review `README.md` for detailed info
3. Check `DATA_FORMAT_GUIDE.md` for date format help
4. Verify your data format (YYYYMM is recommended)
5. Try with `sample_data.csv` to isolate issues
6. Check error messages in the app

---

**Ready to forecast? Upload your YYYYMM data and let's go! 🚀**
