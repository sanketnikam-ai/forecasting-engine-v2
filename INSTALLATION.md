# Installation Guide

## ✅ Recommended: Core Installation (22 Models)

This installs all the essential models and works reliably on Streamlit Cloud.

```bash
cd forecasting_app
pip install -r requirements.txt
```

**Includes:**
- ✅ All Statistical Models (12)
- ✅ All Machine Learning Models (7)
- ✅ Facebook Prophet
- ✅ Ensemble
- ✅ **Total: 22 reliable models**

## 🎯 Optional: Advanced Models (+3 Models)

Only install if you specifically need Neural Prophet or LSTM:

```bash
pip install -r requirements-optional.txt
```

**Adds:**
- Neural Prophet (requires PyTorch - ~500MB)
- LSTM (requires TensorFlow - ~400MB)

**Warning:** These are heavy dependencies and may fail on some platforms.

## 🚀 Quick Start

### Option 1: Standard Installation (Recommended)
```bash
# Clone/download the app
cd forecasting_app

# Install core dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Option 2: With Optional Models
```bash
# Core installation first
pip install -r requirements.txt

# Then optional (if needed)
pip install -r requirements-optional.txt

# Run
streamlit run app.py
```

## 📊 What You Get

### With Core Installation:
- Naive Forecast ✅
- Seasonal Naive ✅
- Simple Moving Average ✅
- Weighted Moving Average ✅
- Exponential Smoothing ✅
- Holt-Winters ✅
- ARIMA ✅
- SARIMA ✅
- Auto ARIMA ✅
- TBATS ✅
- Linear Regression ✅
- Random Forest ✅
- XGBoost ✅
- LightGBM ✅
- CatBoost ✅
- Gradient Boosting ✅
- Facebook Prophet ✅
- Ensemble ✅

### With Optional Installation:
- Everything above +
- Neural Prophet ⭐
- LSTM ⭐

## 🔧 Platform-Specific Notes

### Streamlit Cloud:
```bash
# Use only: requirements.txt
# Do NOT use requirements-optional.txt
```

### Local Development:
```bash
# Use both if you want all features
pip install -r requirements.txt
pip install -r requirements-optional.txt
```

### Heroku:
```bash
# Use only: requirements.txt
# Optional models may exceed memory limits
```

## ❓ FAQ

**Q: Do I need to install optional models?**
A: No! The app works great with just core models (22 algorithms).

**Q: What happens if I don't install optional models?**
A: They're automatically disabled in the UI. No errors, no problems.

**Q: Which should I use for production?**
A: Use `requirements.txt` only for reliable, fast deployment.

**Q: Installation is failing, what do I do?**
A: Try installing core models only first:
```bash
pip install -r requirements.txt
```

**Q: Can I install just one optional model?**
A: Yes! Install manually:
```bash
# For Neural Prophet
pip install neuralprophet

# For LSTM
pip install tensorflow keras
```

## ✅ Verification

After installation, run:
```bash
streamlit run app.py
```

You should see:
- App loads successfully ✅
- Can upload CSV ✅
- Models appear in sidebar ✅
- Optional models show as disabled (if not installed) ✅

## 🆘 Getting Help

If installation fails:
1. Try core installation only
2. Check Python version (3.8+ required)
3. Update pip: `pip install --upgrade pip`
4. Install one package at a time to find the problem

---

**Recommended: Start with core installation - it includes 22 powerful models!** 🚀
