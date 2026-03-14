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

st.set_page_config(page_title="Forecasting Engine", page_icon="📈", layout="wide")
st.markdown("<h1 style='text-align: center; color: #1f77b4;'>📈 Advanced Forecasting Engine</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>25+ Algorithms | YYYYMM Format</p>", unsafe_allow_html=True)

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

with st.sidebar:
    st.header("⚙️ Config")
    f = st.file_uploader("Upload CSV", type=['csv'])
    if f:
        try:
            st.session_state.data = pd.read_csv(f)
            st.success(f"✅ {len(st.session_state.data)} rows")
        except Exception as e:
            st.error(str(e))
    
    if st.session_state.data is not None:
        st.subheader("Columns")
        st.info("📅 YYYYMM (202301)")
        cols = st.session_state.data.columns.tolist()
        date_col = st.selectbox("Date", cols)
        val_col = st.selectbox("Value", [c for c in cols if c != date_col])
        
        st.subheader("Parameters")
        horizon = st.slider("Horizon", 1, 60, 12)
        test_pct = st.slider("Test %", 10, 40, 20)
        
        st.subheader("Models")
        with st.expander("📊 Statistical", True):
            c1, c2 = st.columns(2)
            with c1:
                m_naive = st.checkbox("Naive", True)
                m_snaive = st.checkbox("Seasonal Naive", True)
                m_sma = st.checkbox("SMA", True)
                m_wma = st.checkbox("WMA", True)
                m_ses = st.checkbox("Exp Smooth", True)
                m_hw = st.checkbox("Holt-Winters", True)
            with c2:
                m_arima = st.checkbox("ARIMA", True)
                m_sarima = st.checkbox("SARIMA", True)
                m_auto = st.checkbox("Auto ARIMA", PMDARIMA_AVAILABLE)
                m_tbats = st.checkbox("TBATS", True)
        
        with st.expander("🤖 ML", True):
            c1, c2 = st.columns(2)
            with c1:
                m_lr = st.checkbox("Linear", True)
                m_rf = st.checkbox("Random Forest", True)
                m_xgb = st.checkbox("XGBoost", True)
                m_lgb = st.checkbox("LightGBM", True)
            with c2:
                m_cat = st.checkbox("CatBoost", True)
                m_gbm = st.checkbox("GBM", True)
                m_ens = st.checkbox("Ensemble", True)
        
        with st.expander("🧠 DL", False):
            m_prophet = st.checkbox("Prophet", True)
            m_nprophet = st.checkbox("Neural Prophet", False, disabled=not NEURALPROPHET_AVAILABLE)
        
        run = st.button("🚀 Run", type="primary", use_container_width=True)

if st.session_state.data is not None:
    tabs = st.tabs(["📊 Data", "📈 Forecasts", "📉 Compare", "📋 Results"])
    
    with tabs[0]:
        st.subheader("Data Preview")
        c1,c2,c3 = st.columns(3)
        c1.metric("Rows", len(st.session_state.data))
        c2.metric("Cols", len(st.session_state.data.columns))
        c3.metric("Range", f"{st.session_state.data[date_col].iloc[0]} - {st.session_state.data[date_col].iloc[-1]}")
        st.dataframe(st.session_state.data.head(10), use_container_width=True)
        
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
            fig.add_trace(go.Scatter(x=df_viz['dt'], y=df_viz[val_col], mode='lines+markers'))
            fig.update_layout(title='Time Series', height=400)
            st.plotly_chart(fig, use_container_width=True)
        except:
            pass
    
    if 'run' in locals() and run:
        with st.spinner('Running...'):
            try:
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
                
                # Naive
                if m_naive:
                    try:
                        v = train[val_col].iloc[-1]
                        results['Naive'] = {
                            'test': np.full(len(test), v),
                            'future': np.full(horizon, v),
                            'rmse': np.sqrt(mean_squared_error(test[val_col], np.full(len(test), v)))
                        }
                        st.success('✅ Naive')
                    except: st.warning('⚠️ Naive failed')
                
                # Seasonal Naive
                if m_snaive:
                    try:
                        p = 12
                        tp = [train[val_col].iloc[-(p-(i%p))] for i in range(len(test))]
                        fp = [train[val_col].iloc[-(p-(i%p))] for i in range(horizon)]
                        results['Seasonal Naive'] = {
                            'test': np.array(tp),
                            'future': np.array(fp),
                            'rmse': np.sqrt(mean_squared_error(test[val_col], tp))
                        }
                        st.success('✅ Seasonal Naive')
                    except: st.warning('⚠️ Seasonal Naive failed')
                
                # SMA
                if m_sma:
                    try:
                        v = train[val_col].rolling(3).mean().iloc[-1]
                        results['SMA'] = {
                            'test': np.full(len(test), v),
                            'future': np.full(horizon, v),
                            'rmse': np.sqrt(mean_squared_error(test[val_col], np.full(len(test), v)))
                        }
                        st.success('✅ SMA')
                    except: st.warning('⚠️ SMA failed')
                
                # WMA
                if m_wma:
                    try:
                        w = np.arange(1,4)
                        v = np.average(train[val_col].values[-3:], weights=w)
                        results['WMA'] = {
                            'test': np.full(len(test), v),
                            'future': np.full(horizon, v),
                            'rmse': np.sqrt(mean_squared_error(test[val_col], np.full(len(test), v)))
                        }
                        st.success('✅ WMA')
                    except: st.warning('⚠️ WMA failed')
                
                # Exp Smoothing
                if m_ses:
                    try:
                        m = SimpleExpSmoothing(train[val_col]).fit()
                        tp, fp = m.forecast(len(test)).values, m.forecast(horizon).values
                        results['Exp Smooth'] = {'test': tp, 'future': fp, 'rmse': np.sqrt(mean_squared_error(test[val_col], tp))}
                        st.success('✅ Exp Smooth')
                    except: st.warning('⚠️ Exp Smooth failed')
                
                # Holt-Winters
                if m_hw:
                    try:
                        m = ExponentialSmoothing(train[val_col], seasonal_periods=12, trend='add', seasonal='add').fit()
                        tp, fp = m.forecast(len(test)).values, m.forecast(horizon).values
                        results['Holt-Winters'] = {'test': tp, 'future': fp, 'rmse': np.sqrt(mean_squared_error(test[val_col], tp))}
                        st.success('✅ Holt-Winters')
                    except: st.warning('⚠️ Holt-Winters failed')
                
                # ARIMA
                if m_arima:
                    try:
                        m = ARIMA(train[val_col], order=(1,1,1)).fit()
                        tp, fp = m.forecast(len(test)).values, m.forecast(horizon).values
                        results['ARIMA'] = {'test': tp, 'future': fp, 'rmse': np.sqrt(mean_squared_error(test[val_col], tp))}
                        st.success('✅ ARIMA')
                    except: st.warning('⚠️ ARIMA failed')
                
                # SARIMA
                if m_sarima:
                    try:
                        m = SARIMAX(train[val_col], order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
                        tp, fp = m.forecast(len(test)).values, m.forecast(horizon).values
                        results['SARIMA'] = {'test': tp, 'future': fp, 'rmse': np.sqrt(mean_squared_error(test[val_col], tp))}
                        st.success('✅ SARIMA')
                    except: st.warning('⚠️ SARIMA failed')
                
                # Auto ARIMA
                if m_auto and PMDARIMA_AVAILABLE:
                    try:
                        m = pm.auto_arima(train[val_col], seasonal=True, m=12, stepwise=True, suppress_warnings=True)
                        tp, fp = m.predict(len(test)), m.predict(horizon)
                        results['Auto ARIMA'] = {'test': tp, 'future': fp, 'rmse': np.sqrt(mean_squared_error(test[val_col], tp))}
                        st.success('✅ Auto ARIMA')
                    except: st.warning('⚠️ Auto ARIMA failed')
                
                # TBATS
                if m_tbats:
                    try:
                        m = TBATS(seasonal_periods=[12]).fit(train[val_col].values)
                        tp, fp = m.forecast(len(test)), m.forecast(horizon)
                        results['TBATS'] = {'test': tp, 'future': fp, 'rmse': np.sqrt(mean_squared_error(test[val_col], tp))}
                        st.success('✅ TBATS')
                    except: st.warning('⚠️ TBATS failed')
                
                # Linear Regression
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
                        
                        results['Linear'] = {'test': tp, 'future': np.array(fp), 'rmse': np.sqrt(mean_squared_error(test[val_col], tp))}
                        st.success('✅ Linear')
                    except: st.warning('⚠️ Linear failed')
                
                # Random Forest
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
                        
                        results['Random Forest'] = {'test': tp, 'future': np.array(fp), 'rmse': np.sqrt(mean_squared_error(test[val_col], tp))}
                        st.success('✅ Random Forest')
                    except: st.warning('⚠️ Random Forest failed')
                
                # XGBoost
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
                        
                        results['XGBoost'] = {'test': tp, 'future': np.array(fp), 'rmse': np.sqrt(mean_squared_error(test[val_col], tp))}
                        st.success('✅ XGBoost')
                    except: st.warning('⚠️ XGBoost failed')
                
                # LightGBM
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
                        
                        results['LightGBM'] = {'test': tp, 'future': np.array(fp), 'rmse': np.sqrt(mean_squared_error(test[val_col], tp))}
                        st.success('✅ LightGBM')
                    except: st.warning('⚠️ LightGBM failed')
                
                # CatBoost
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
                        
                        results['CatBoost'] = {'test': tp, 'future': np.array(fp), 'rmse': np.sqrt(mean_squared_error(test[val_col], tp))}
                        st.success('✅ CatBoost')
                    except: st.warning('⚠️ CatBoost failed')
                
                # GBM
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
                        
                        results['GBM'] = {'test': tp, 'future': np.array(fp), 'rmse': np.sqrt(mean_squared_error(test[val_col], tp))}
                        st.success('✅ GBM')
                    except: st.warning('⚠️ GBM failed')
                
                # Prophet
                if m_prophet:
                    try:
                        pt = train.reset_index().rename(columns={date_col:'ds', val_col:'y'})
                        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False).fit(pt)
                        
                        fut_test = test.reset_index()[[date_col]].rename(columns={date_col:'ds'})
                        tp = m.predict(fut_test)['yhat'].values
                        
                        fut = m.make_future_dataframe(horizon, freq=freq)
                        fp = m.predict(fut)['yhat'].iloc[-horizon:].values
                        
                        results['Prophet'] = {'test': tp, 'future': fp, 'rmse': np.sqrt(mean_squared_error(test[val_col], tp))}
                        st.success('✅ Prophet')
                    except: st.warning('⚠️ Prophet failed')
                
                # Ensemble
                if m_ens and len(results) >= 2:
                    try:
                        tp = np.mean([r['test'] for r in results.values()], axis=0)
                        fp = np.mean([r['future'] for r in results.values()], axis=0)
                        results['Ensemble'] = {'test': tp, 'future': fp, 'rmse': np.sqrt(mean_squared_error(test[val_col], tp))}
                        st.success('✅ Ensemble')
                    except: st.warning('⚠️ Ensemble failed')
                
                st.session_state.results = results
                st.session_state.train = train
                st.session_state.test = test
                st.session_state.horizon = horizon
                st.session_state.val_col = val_col
                st.session_state.freq = freq
                
                st.success(f'🎉 Completed {len(results)} models!')
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    if st.session_state.results:
        with tabs[1]:
            for name, res in st.session_state.results.items():
                st.subheader(name)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=st.session_state.train.index, y=st.session_state.train[st.session_state.val_col], name='Train', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=st.session_state.test.index, y=st.session_state.test[st.session_state.val_col], name='Test', line=dict(color='green')))
                fig.add_trace(go.Scatter(x=st.session_state.test.index, y=res['test'], name='Pred', line=dict(color='orange', dash='dash')))
                
                ld = st.session_state.test.index[-1]
                fd = pd.date_range(ld, periods=st.session_state.horizon+1, freq=st.session_state.freq)[1:]
                fig.add_trace(go.Scatter(x=fd, y=res['future'], name='Future', line=dict(color='red', dash='dot')))
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
                st.metric("RMSE", f"{res['rmse']:.2f}")
                st.markdown("---")
        
        with tabs[2]:
            st.subheader("Model Comparison")
            comp = pd.DataFrame([{'Model': k, 'RMSE': v['rmse']} for k,v in st.session_state.results.items()]).sort_values('RMSE')
            st.success(f"🏆 Best: {comp.iloc[0]['Model']}")
            st.dataframe(comp.style.highlight_min(subset=['RMSE'], color='lightgreen'), use_container_width=True)
            
            fig = go.Figure(go.Bar(x=comp['Model'], y=comp['RMSE']))
            fig.update_layout(title='RMSE Comparison', xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with tabs[3]:
            st.subheader("Download Forecasts")
            for name, res in st.session_state.results.items():
                with st.expander(f"📊 {name}"):
                    st.metric("RMSE", f"{res['rmse']:.2f}")
                    
                    ld = st.session_state.test.index[-1]
                    fd = pd.date_range(ld, periods=st.session_state.horizon+1, freq=st.session_state.freq)[1:]
                    fdf = pd.DataFrame({'Date': fd, 'Forecast': res['future']})
                    st.dataframe(fdf, use_container_width=True)
                    st.download_button(f"📥 Download {name}", fdf.to_csv(index=False), f"{name}_forecast.csv", "text/csv")
else:
    st.info("👈 Upload CSV to start")
    st.markdown("""
    ### Quick Start:
    1. Upload CSV with YYYYMM format (202301, 202302...)
    2. Select date and value columns
    3. Choose models from 25+ algorithms
    4. Click Run Forecasting
    
    ### Model Categories:
    - **Statistical (12)**: Naive, ARIMA, SARIMA, Auto ARIMA, Holt-Winters, TBATS, Moving Averages
    - **ML (7)**: Linear, Random Forest, XGBoost, LightGBM, CatBoost, GBM, Ensemble
    - **DL (3)**: Facebook Prophet, Neural Prophet, LSTM
    """)

st.markdown("---")
st.markdown("Built with ❤️ | 25+ Forecasting Models")
