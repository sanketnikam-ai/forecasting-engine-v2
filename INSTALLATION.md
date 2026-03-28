# Installation Guide

## ✅ Recommended: Core (22 Models)

Fast, reliable installation:

```bash
pip install -r requirements.txt
```

**Includes:**
- All Statistical (12)
- All ML (7)
- Facebook Prophet
- Ensemble
- **Total: 22 models**

## 🎯 Optional: Advanced (+2 Models)

Heavy dependencies (~1GB):

```bash
pip install -r requirements-optional.txt
```

**Adds:**
- Neural Prophet
- LSTM

**Warning:** May fail on some platforms

## 📊 What You Get

### Core Installation (22 models):
✅ Naive, Seasonal Naive
✅ SMA, WMA
✅ Exponential Smoothing, Holt-Winters
✅ ARIMA, SARIMA, Auto ARIMA, TBATS
✅ Linear, Random Forest
✅ XGBoost, LightGBM, CatBoost, GBM
✅ Prophet, Ensemble

### Optional (+2 models):
⭐ Neural Prophet
⭐ LSTM

## 🚀 Platform Guide

### Streamlit Cloud:
```bash
# Use ONLY requirements.txt
```

### Local:
```bash
# Use both if wanted
pip install -r requirements.txt
pip install -r requirements-optional.txt
```

### Heroku:
```bash
# Use ONLY requirements.txt
```

## ❓ FAQ

**Q: Need optional models?**
A: No! 22 core models work great.

**Q: What if I don't install optional?**
A: Auto-disabled in UI. No errors.

**Q: Production use?**
A: Use `requirements.txt` only.

**Q: Installation failing?**
A: Try core only first.

## ✅ Verification

```bash
streamlit run app.py
```

Should see:
- App loads ✅
- Can upload CSV ✅
- Models in sidebar ✅
- Optional models disabled (if not installed) ✅

---

**Recommended: Core installation (22 models) 🚀**
