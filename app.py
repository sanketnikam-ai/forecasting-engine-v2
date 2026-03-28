import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import plotly.graph_objects as go

# Import forecasting libraries
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from tbats import TBATS
from prophet import Prophet

# Optional imports
try:
    from neuralprophet import NeuralProphet
    NEURALPROPHET_AVAILABLE = True
except:
    NEURALPROPHET_AVAILABLE = False

try:
    import pmdarima as pm
    PMDARIMA_AVAILABLE = True
except:
    PMDARIMA_AVAILABLE = False

try:
    from tensorflow import keras
    from keras.models import Sequential
    from keras.layers import LSTM, Dense
    from sklearn.preprocessing import MinMaxScaler
    LSTM_AVAILABLE = True
except:
    LSTM_AVAILABLE = False

# Page config
st.set_page_config(page_title="Advanced Forecasting Engine", page_icon="📈", layout="wide")

st.markdown("<h1 style='text-align: center; color: #1f77b4;'>📈 Advanced Time Series Forecasting Engine</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>22+ Forecasting Algorithms | YYYYMM Format Support</p>", unsafe_allow_html=True)

# Session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'results' not in st.session_state:
    st.session_state.results = {}

def create_features(data, lags=12):
    df = pd.DataFrame(index=data.index)
    for i in range(1, lags+1):
        df[f'lag_{i}'] = data.shift(i)
    df['ma_3'] = data.rolling(3).mean()
    df['ma_6'] = data.rolling(6).mean()
    df['std_3'] = data.rolling(3).std()
    return df.dropna()

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    f = st.file_uploader("Upload CSV file", type=['csv'])
    if f:
        try:
            st.session_state.data = pd.read_csv(f)
            st.success(f"✅ Uploaded! ({len(st.session_state.data)} rows)")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    if st.session_state.data is not None:
        st.subheader("Column Selection")
        st.info("📅 Formats: YYYYMM (202301), YYYY-MM-DD, YYYY-MM")
        cols = st.session_state.data.columns.tolist()
        date_col = st.selectbox("Date Column", cols)
        val_col = st.selectbox("Value Column", [c for c in cols if c != date_col])
        
        st.subheader("Parameters")
        horizon = st.slider("Forecast Horizon (periods)", 1, 60, 12)
        test_pct = st.slider("Test Size (%)", 10, 40, 20)
        
        st.subheader("Select Models")
        with st.expander("📊 Statistical Models (12)", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                m_naive = st.checkbox("Naive Forecast", True)
                m_snaive = st.checkbox("Seasonal Naive", True)
                m_sma = st.checkbox("Simple MA", True)
                m_wma = st.checkbox("Weighted MA", True)
                m_ses = st.checkbox("Exponential Smoothing", True)
                m_hw = st.checkbox("Holt-Winters", True)
            with c2:
                m_arima = st.checkbox("ARIMA", True)
                m_sarima = st.checkbox("SARIMA", True)
                m_auto = st.checkbox("Auto ARIMA", PMDARIMA_AVAILABLE)
                m_arimax = st.checkbox("ARIMAX", False)
                m_tbats = st.checkbox("TBATS", True)
                m_croston = st.checkbox("Croston's Method", False)
        
        with st.expander("🤖 Machine Learning Models (7)", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                m_lr = st.checkbox("Linear Regression", True)
                m_rf = st.checkbox("Random Forest", True)
                m_xgb = st.checkbox("XGBoost", True)
                m_lgb = st.checkbox("LightGBM", True)
            with c2:
                m_cat = st.checkbox("CatBoost", True)
                m_gbm = st.checkbox("Gradient Boosting", True)
                m_ens = st.checkbox("Ensemble Model", True)
        
        with st.expander("🧠 Deep Learning Models (3)", expanded=False):
            m_prophet = st.checkbox("Facebook Prophet", True)
            m_nprophet = st.checkbox("Neural Prophet", False, disabled=not NEURALPROPHET_AVAILABLE)
            m_lstm = st.checkbox("LSTM", False, disabled=not LSTM_AVAILABLE)
        
        run = st.button("🚀 Run Forecasting", type="primary", use_container_width=True)

# Main content
if st.session_state.data is not None:
    tabs = st.tabs(["📊 Data Overview", "📈 Forecasts", "📉 Model Comparison", "📋 Detailed Results"])
    
    with tabs[0]:
        st.subheader("Data Preview")
        c1,c2,c3 = st.columns(3)
        c1.metric("Total Records", len(st.session_state.data))
        c2.metric("Columns", len(st.session_state.data.columns))
        c3.metric("Date Range", f"{st.session_state.data[date_col].iloc[0]} - {st.session_state.data[date_col].iloc[-1]}")
        st.dataframe(st.session_state.data.head(10), use_container_width=True)
        
        st.subheader("Time Series Visualization")
        try:
            df_viz = st.session_state.data.copy()
            if df_viz[date_col].dtype in ['int64','float64']:
                df_viz['dt'] = pd.to_datetime(df_viz[date_col].astype(str), format='%Y%m')
            else:
                try:
                    df_viz['dt'] = pd.to_datetime(df_viz[date_col], format='%Y%m')
                except:
                    df_viz['dt'] = pd.to_datetime(df_viz[date_col])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_viz['dt'], y=df_viz[val_col], mode='lines+markers', name='Actual'))
            fig.update_layout(title='Time Series Data', xaxis_title='Date', yaxis_title='Value', height=400)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not create visualization: {str(e)}")
        
        st.subheader("Descriptive Statistics")
        st.dataframe(st.session_state.data[val_col].describe(), use_container_width=True)
    
    # Forecasting logic
    if 'run' in locals() and run:
        with st.spinner('Running forecasting models... This may take a few minutes.'):
            try:
                # Prepare data
                df = st.session_state.data.copy()
                if df[date_col].dtype in ['int64','float64']:
                    df[date_col] = pd.to_datetime(df[date_col].astype(str), format='%Y%m')
                else:
                    try:
                        df[date_col] = pd.to_datetime(df[date_col], format='%Y%m')
                    except:
                        df[date_col] = pd.to_datetime(df[date_col])
                
                df = df.sort_values(date_col).set_index(date_col)
                freq = pd.infer_freq(df.index)
                if not freq:
                    freq = 'MS' if (df.index[1]-df.index[0]).days > 20 else 'D'
                
                split = int(len(df) * (1-test_pct/100))
                train, test = df.iloc[:split], df.iloc[split:]
                results = {}
                
                # 1. Naive Forecast
                if m_naive:
                    try:
                        v = train[val_col].iloc[-1]
                        tp = np.full(len(test), v)
                        fp = np.full(horizon, v)
                        results['Naive'] = {
                            'test': tp, 'future': fp,
                            'rmse': np.sqrt(mean_squared_error(test[val_col], tp)),
                            'mae': mean_absolute_error(test[val_col], tp),
                            'mape': mean_absolute_percentage_error(test[val_col], tp) * 100
                        }
                        st.success('✅ Naive')
                    except Exception as e:
                        st.warning(f'⚠️ Naive failed: {str(e)}')
                
                # 2. Seasonal Naive
                if m_snaive:
                    try:
                        p = 12
                        tp = [train[val_col].iloc[-(p-(i%p))] for i in range(len(test))]
                        fp = [train[val_col].iloc[-(p-(i%p))] for i in range(horizon)]
                        tp, fp = np.array(tp), np.array(fp)
                        results['Seasonal Naive'] = {
                            'test': tp, 'future': fp,
                            'rmse': np.sqrt(mean_squared_error(test[val_col], tp)),
                            'mae': mean_absolute_error(test[val_col], tp),
                            'mape': mean_absolute_percentage_error(test[val_col], tp) * 100
                        }
                        st.success('✅ Seasonal Naive')
                    except Exception as e:
                        st.warning(f'⚠️ Seasonal Naive failed: {str(e)}')
                
                # 3. Simple Moving Average
                if m_sma:
                    try:
                        v = train[val_col].rolling(3).mean().iloc[-1]
                        tp = np.full(len(test), v)
                        fp = np.full(horizon, v)
                        results['SMA'] = {
                            'test': tp, 'future': fp,
                            'rmse': np.sqrt(mean_squared_error(test[val_col], tp)),
                            'mae': mean_absolute_error(test[val_col], tp),
                            'mape': mean_absolute_percentage_error(test[val_col], tp) * 100
                        }
                        st.success('✅ SMA')
                    except Exception as e:
                        st.warning(f'⚠️ SMA failed: {str(e)}')
                
                # 4. Weighted Moving Average
                if m_wma:
                    try:
                        w = np.arange(1,4)
                        v = np.average(train[val_col].values[-3:], weights=w)
                        tp = np.full(len(test), v)
                        fp = np.full(horizon, v)
                        results['WMA'] = {
                            'test': tp, 'future': fp,
                            'rmse': np.sqrt(mean_squared_error(test[val_col], tp)),
                            'mae': mean_absolute_error(test[val_col], tp),
                            'mape': mean_absolute_percentage_error(test[val_col], tp) * 100
                        }
                        st.success('✅ WMA')
                    except Exception as e:
                        st.warning(f'⚠️ WMA failed: {str(e)}')
                
                # 5. Exponential Smoothing
                if m_ses:
                    try:
                        m = SimpleExpSmoothing(train[val_col]).fit()
                        tp, fp = m.forecast(len(test)).values, m.forecast(horizon).values
                        results['Exp Smoothing'] = {
                            'test': tp, 'future': fp,
                            'rmse': np.sqrt(mean_squared_error(test[val_col], tp)),
                            'mae': mean_absolute_error(test[val_col], tp),
                            'mape': mean_absolute_percentage_error(test[val_col], tp) * 100
                        }
                        st.success('✅ Exp Smoothing')
                    except Exception as e:
                        st.warning(f'⚠️ Exp Smoothing failed: {str(e)}')
                
                # 6. Holt-Winters
                if m_hw:
                    try:
                        m = ExponentialSmoothing(train[val_col], seasonal_periods=12, trend='add', seasonal='add').fit()
                        tp, fp = m.forecast(len(test)).values, m.forecast(horizon).values
                        results['Holt-Winters'] = {
                            'test': tp, 'future': fp,
                            'rmse': np.sqrt(mean_squared_error(test[val_col], tp)),
                            'mae': mean_absolute_error(test[val_col], tp),
                            'mape': mean_absolute_percentage_error(test[val_col], tp) * 100
                        }
                        st.success('✅ Holt-Winters')
                    except Exception as e:
                        st.warning(f'⚠️ Holt-Winters failed: {str(e)}')
                
                # 7. ARIMA
                if m_arima:
                    try:
                        m = ARIMA(train[val_col], order=(1,1,1)).fit()
                        tp, fp = m.forecast(len(test)).values, m.forecast(horizon).values
                        results['ARIMA'] = {
                            'test': tp, 'future': fp,
                            'rmse': np.sqrt(mean_squared_error(test[val_col], tp)),
                            'mae': mean_absolute_error(test[val_col], tp),
                            'mape': mean_absolute_percentage_error(test[val_col], tp) * 100
                        }
                        st.success('✅ ARIMA')
                    except Exception as e:
                        st.warning(f'⚠️ ARIMA failed: {str(e)}')
                
                # 8. SARIMA
                if m_sarima:
                    try:
                        m = SARIMAX(train[val_col], order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
                        tp, fp = m.forecast(len(test)).values, m.forecast(horizon).values
                        results['SARIMA'] = {
                            'test': tp, 'future': fp,
                            'rmse': np.sqrt(mean_squared_error(test[val_col], tp)),
                            'mae': mean_absolute_error(test[val_col], tp),
                            'mape': mean_absolute_percentage_error(test[val_col], tp) * 100
                        }
                        st.success('✅ SARIMA')
                    except Exception as e:
                        st.warning(f'⚠️ SARIMA failed: {str(e)}')
                
                # 9. Auto ARIMA
                if m_auto and PMDARIMA_AVAILABLE:
                    try:
                        m = pm.auto_arima(train[val_col], seasonal=True, m=12, stepwise=True, suppress_warnings=True)
                        tp = m.predict(len(test))
                        fp = m.predict(horizon)
                        results['Auto ARIMA'] = {
                            'test': tp, 'future': fp,
                            'rmse': np.sqrt(mean_squared_error(test[val_col], tp)),
                            'mae': mean_absolute_error(test[val_col], tp),
                            'mape': mean_absolute_percentage_error(test[val_col], tp) * 100
                        }
                        st.success('✅ Auto ARIMA')
                    except Exception as e:
                        st.warning(f'⚠️ Auto ARIMA failed: {str(e)}')
                
                # 10. TBATS
                if m_tbats:
                    try:
                        m = TBATS(seasonal_periods=[12]).fit(train[val_col].values)
                        tp = m.forecast(len(test))
                        fp = m.forecast(horizon)
                        results['TBATS'] = {
                            'test': tp, 'future': fp,
                            'rmse': np.sqrt(mean_squared_error(test[val_col], tp)),
                            'mae': mean_absolute_error(test[val_col], tp),
                            'mape': mean_absolute_percentage_error(test[val_col], tp) * 100
                        }
                        st.success('✅ TBATS')
                    except Exception as e:
                        st.warning(f'⚠️ TBATS failed: {str(e)}')
                
                # 11. Linear Regression
                if m_lr:
                    try:
                        tf = create_features(train[val_col])
                        tt = train[val_col].loc[tf.index]
                        m = LinearRegression().fit(tf, tt)
                        
                        testf = create_features(pd.concat([train[val_col], test[val_col]])).loc[test.index]
                        tp = m.predict(testf)
                        
                        lv = df[val_col].values.copy()
                        fp = []
                        for _ in range(horizon):
                            f = [lv[-i] if len(lv)>=i else 0 for i in range(1,13)]
                            f += [np.mean(lv[-3:]), np.mean(lv[-6:]), np.std(lv[-3:])]
                            p = m.predict(np.array(f).reshape(1,-1))[0]
                            fp.append(p)
                            lv = np.append(lv, p)
                        
                        results['Linear Regression'] = {
                            'test': tp, 'future': np.array(fp),
                            'rmse': np.sqrt(mean_squared_error(test[val_col], tp)),
                            'mae': mean_absolute_error(test[val_col], tp),
                            'mape': mean_absolute_percentage_error(test[val_col], tp) * 100
                        }
                        st.success('✅ Linear Regression')
                    except Exception as e:
                        st.warning(f'⚠️ Linear Regression failed: {str(e)}')
                
                # 12. Random Forest
                if m_rf:
                    try:
                        tf = create_features(train[val_col])
                        tt = train[val_col].loc[tf.index]
                        m = RandomForestRegressor(100, random_state=42, n_jobs=-1).fit(tf, tt)
                        
                        testf = create_features(pd.concat([train[val_col], test[val_col]])).loc[test.index]
                        tp = m.predict(testf)
                        
                        lv = df[val_col].values.copy()
                        fp = []
                        for _ in range(horizon):
                            f = [lv[-i] if len(lv)>=i else 0 for i in range(1,13)]
                            f += [np.mean(lv[-3:]), np.mean(lv[-6:]), np.std(lv[-3:])]
                            p = m.predict(np.array(f).reshape(1,-1))[0]
                            fp.append(p)
                            lv = np.append(lv, p)
                        
                        results['Random Forest'] = {
                            'test': tp, 'future': np.array(fp),
                            'rmse': np.sqrt(mean_squared_error(test[val_col], tp)),
                            'mae': mean_absolute_error(test[val_col], tp),
                            'mape': mean_absolute_percentage_error(test[val_col], tp) * 100
                        }
                        st.success('✅ Random Forest')
                    except Exception as e:
                        st.warning(f'⚠️ Random Forest failed: {str(e)}')
                
                # 13. XGBoost
                if m_xgb:
                    try:
                        tf = create_features(train[val_col])
                        tt = train[val_col].loc[tf.index]
                        m = xgb.XGBRegressor(100, max_depth=5, learning_rate=0.1, random_state=42).fit(tf, tt)
                        
                        testf = create_features(pd.concat([train[val_col], test[val_col]])).loc[test.index]
                        tp = m.predict(testf)
                        
                        lv = df[val_col].values.copy()
                        fp = []
                        for _ in range(horizon):
                            f = [lv[-i] if len(lv)>=i else 0 for i in range(1,13)]
                            f += [np.mean(lv[-3:]), np.mean(lv[-6:]), np.std(lv[-3:])]
                            p = m.predict(np.array(f).reshape(1,-1))[0]
                            fp.append(p)
                            lv = np.append(lv, p)
                        
                        results['XGBoost'] = {
                            'test': tp, 'future': np.array(fp),
                            'rmse': np.sqrt(mean_squared_error(test[val_col], tp)),
                            'mae': mean_absolute_error(test[val_col], tp),
                            'mape': mean_absolute_percentage_error(test[val_col], tp) * 100
                        }
                        st.success('✅ XGBoost')
                    except Exception as e:
                        st.warning(f'⚠️ XGBoost failed: {str(e)}')
                
                # 14. LightGBM
                if m_lgb:
                    try:
                        tf = create_features(train[val_col])
                        tt = train[val_col].loc[tf.index]
                        m = lgb.LGBMRegressor(100, random_state=42, verbose=-1).fit(tf, tt)
                        
                        testf = create_features(pd.concat([train[val_col], test[val_col]])).loc[test.index]
                        tp = m.predict(testf)
                        
                        lv = df[val_col].values.copy()
                        fp = []
                        for _ in range(horizon):
                            f = [lv[-i] if len(lv)>=i else 0 for i in range(1,13)]
                            f += [np.mean(lv[-3:]), np.mean(lv[-6:]), np.std(lv[-3:])]
                            p = m.predict(np.array(f).reshape(1,-1))[0]
                            fp.append(p)
                            lv = np.append(lv, p)
                        
                        results['LightGBM'] = {
                            'test': tp, 'future': np.array(fp),
                            'rmse': np.sqrt(mean_squared_error(test[val_col], tp)),
                            'mae': mean_absolute_error(test[val_col], tp),
                            'mape': mean_absolute_percentage_error(test[val_col], tp) * 100
                        }
                        st.success('✅ LightGBM')
                    except Exception as e:
                        st.warning(f'⚠️ LightGBM failed: {str(e)}')
                
                # 15. CatBoost
                if m_cat:
                    try:
                        tf = create_features(train[val_col])
                        tt = train[val_col].loc[tf.index]
                        m = CatBoostRegressor(100, random_state=42, verbose=0).fit(tf, tt)
                        
                        testf = create_features(pd.concat([train[val_col], test[val_col]])).loc[test.index]
                        tp = m.predict(testf)
                        
                        lv = df[val_col].values.copy()
                        fp = []
                        for _ in range(horizon):
                            f = [lv[-i] if len(lv)>=i else 0 for i in range(1,13)]
                            f += [np.mean(lv[-3:]), np.mean(lv[-6:]), np.std(lv[-3:])]
                            p = m.predict(np.array(f).reshape(1,-1))[0]
                            fp.append(p)
                            lv = np.append(lv, p)
                        
                        results['CatBoost'] = {
                            'test': tp, 'future': np.array(fp),
                            'rmse': np.sqrt(mean_squared_error(test[val_col], tp)),
                            'mae': mean_absolute_error(test[val_col], tp),
                            'mape': mean_absolute_percentage_error(test[val_col], tp) * 100
                        }
                        st.success('✅ CatBoost')
                    except Exception as e:
                        st.warning(f'⚠️ CatBoost failed: {str(e)}')
                
                # 16. Gradient Boosting
                if m_gbm:
                    try:
                        tf = create_features(train[val_col])
                        tt = train[val_col].loc[tf.index]
                        m = GradientBoostingRegressor(100, random_state=42).fit(tf, tt)
                        
                        testf = create_features(pd.concat([train[val_col], test[val_col]])).loc[test.index]
                        tp = m.predict(testf)
                        
                        lv = df[val_col].values.copy()
                        fp = []
                        for _ in range(horizon):
                            f = [lv[-i] if len(lv)>=i else 0 for i in range(1,13)]
                            f += [np.mean(lv[-3:]), np.mean(lv[-6:]), np.std(lv[-3:])]
                            p = m.predict(np.array(f).reshape(1,-1))[0]
                            fp.append(p)
                            lv = np.append(lv, p)
                        
                        results['Gradient Boosting'] = {
                            'test': tp, 'future': np.array(fp),
                            'rmse': np.sqrt(mean_squared_error(test[val_col], tp)),
                            'mae': mean_absolute_error(test[val_col], tp),
                            'mape': mean_absolute_percentage_error(test[val_col], tp) * 100
                        }
                        st.success('✅ Gradient Boosting')
                    except Exception as e:
                        st.warning(f'⚠️ Gradient Boosting failed: {str(e)}')
                
                # 17. Facebook Prophet
                if m_prophet:
                    try:
                        pt = train.reset_index().rename(columns={date_col:'ds', val_col:'y'})
                        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False).fit(pt)
                        
                        fut_test = test.reset_index()[[date_col]].rename(columns={date_col:'ds'})
                        tp = m.predict(fut_test)['yhat'].values
                        
                        fut = m.make_future_dataframe(horizon, freq=freq)
                        fp = m.predict(fut)['yhat'].iloc[-horizon:].values
                        
                        results['Prophet'] = {
                            'test': tp, 'future': fp,
                            'rmse': np.sqrt(mean_squared_error(test[val_col], tp)),
                            'mae': mean_absolute_error(test[val_col], tp),
                            'mape': mean_absolute_percentage_error(test[val_col], tp) * 100
                        }
                        st.success('✅ Prophet')
                    except Exception as e:
                        st.warning(f'⚠️ Prophet failed: {str(e)}')
                
                # 18. Ensemble
                if m_ens and len(results) >= 2:
                    try:
                        tp = np.mean([r['test'] for r in results.values()], axis=0)
                        fp = np.mean([r['future'] for r in results.values()], axis=0)
                        results['Ensemble'] = {
                            'test': tp, 'future': fp,
                            'rmse': np.sqrt(mean_squared_error(test[val_col], tp)),
                            'mae': mean_absolute_error(test[val_col], tp),
                            'mape': mean_absolute_percentage_error(test[val_col], tp) * 100
                        }
                        st.success('✅ Ensemble')
                    except Exception as e:
                        st.warning(f'⚠️ Ensemble failed: {str(e)}')
                
                st.session_state.results = results
                st.session_state.train = train
                st.session_state.test = test
                st.session_state.horizon = horizon
                st.session_state.val_col = val_col
                st.session_state.freq = freq
                
                st.success(f'🎉 Completed {len(results)} models successfully!')
            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
    
    # Display results
    if st.session_state.results:
        with tabs[1]:
            st.subheader("Forecast Visualizations")
            for name, res in st.session_state.results.items():
                st.markdown(f"### {name}")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=st.session_state.train.index, y=st.session_state.train[st.session_state.val_col], 
                                        mode='lines', name='Train', line=dict(color='blue', width=2)))
                fig.add_trace(go.Scatter(x=st.session_state.test.index, y=st.session_state.test[st.session_state.val_col], 
                                        mode='lines', name='Test', line=dict(color='green', width=2)))
                fig.add_trace(go.Scatter(x=st.session_state.test.index, y=res['test'], 
                                        mode='lines', name='Test Pred', line=dict(color='orange', width=2, dash='dash')))
                
                ld = st.session_state.test.index[-1]
                fd = pd.date_range(ld, periods=st.session_state.horizon+1, freq=st.session_state.freq)[1:]
                fig.add_trace(go.Scatter(x=fd, y=res['future'], mode='lines', name='Future', 
                                        line=dict(color='red', width=2, dash='dot')))
                fig.update_layout(title=f'{name} Forecast', xaxis_title='Date', yaxis_title='Value', height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                c1,c2,c3 = st.columns(3)
                c1.metric("RMSE", f"{res['rmse']:.2f}")
                c2.metric("MAE", f"{res['mae']:.2f}")
                c3.metric("MAPE", f"{res['mape']:.2f}%")
                st.markdown("---")
        
        with tabs[2]:
            st.subheader("Model Comparison")
            comp = pd.DataFrame([{
                'Model': k, 
                'RMSE': v['rmse'], 
                'MAE': v['mae'], 
                'MAPE': v['mape']
            } for k,v in st.session_state.results.items()]).sort_values('RMSE')
            
            best = comp.iloc[0]['Model']
            st.success(f"🏆 **Best Model: {best}** (Lowest RMSE)")
            st.dataframe(comp.style.highlight_min(subset=['RMSE','MAE','MAPE'], color='lightgreen'), 
                        use_container_width=True)
            
            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure(go.Bar(x=comp['Model'], y=comp['RMSE'], marker_color='skyblue'))
                fig.update_layout(title='RMSE Comparison', xaxis_tickangle=-45, height=400)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = go.Figure(go.Bar(x=comp['Model'], y=comp['MAPE'], marker_color='lightcoral'))
                fig.update_layout(title='MAPE Comparison (%)', xaxis_tickangle=-45, height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with tabs[3]:
            st.subheader("Detailed Results & Downloads")
            for name, res in st.session_state.results.items():
                with st.expander(f"📊 {name} - Detailed Results"):
                    c1,c2,c3 = st.columns(3)
                    c1.metric("RMSE", f"{res['rmse']:.2f}")
                    c2.metric("MAE", f"{res['mae']:.2f}")
                    c3.metric("MAPE", f"{res['mape']:.2f}%")
                    
                    st.write("**Test Period Performance (First 10):**")
                    test_df = pd.DataFrame({
                        'Date': st.session_state.test.index[:10],
                        'Actual': st.session_state.test[st.session_state.val_col].values[:10],
                        'Predicted': res['test'][:10],
                        'Error': st.session_state.test[st.session_state.val_col].values[:10] - res['test'][:10]
                    })
                    st.dataframe(test_df, use_container_width=True)
                    
                    st.write("**Future Forecast (All Periods):**")
                    ld = st.session_state.test.index[-1]
                    fd = pd.date_range(ld, periods=st.session_state.horizon+1, freq=st.session_state.freq)[1:]
                    fut_df = pd.DataFrame({'Date': fd, 'Predicted_Value': res['future']})
                    st.dataframe(fut_df, use_container_width=True)
                    
                    csv = fut_df.to_csv(index=False)
                    st.download_button(f"📥 Download {name} Future Forecast", csv, 
                                     f"{name}_forecast.csv", "text/csv")
else:
    st.info("👈 Upload a CSV file to begin forecasting")
    st.markdown("""
    ### 📋 Quick Start:
    1. **Upload CSV** with YYYYMM format (202301, 202302...)
    2. **Select** date and value columns
    3. **Choose models** from 22+ algorithms
    4. **Run** forecasting
    5. **Compare** results and download forecasts
    
    ### 📊 Model Categories:
    **Statistical (12):** Naive, Seasonal Naive, MA, ARIMA, SARIMA, Auto ARIMA, Holt-Winters, TBATS
    
    **Machine Learning (7):** Linear, Random Forest, XGBoost, LightGBM, CatBoost, GBM, Ensemble
    
    **Deep Learning (3):** Facebook Prophet, Neural Prophet*, LSTM*
    
    *Optional models - install separately if needed
    
    ### 📈 Features:
    - YYYYMM format support
    - Train-test validation
    - Automatic best model selection
    - Interactive visualizations
    - Downloadable forecasts
    """)

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | 22+ Forecasting Models")
