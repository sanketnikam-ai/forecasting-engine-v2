import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import forecasting libraries
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from tbats import TBATS
from prophet import Prophet
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="Advanced Forecasting Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="main-header">📈 Advanced Time Series Forecasting Engine</p>', unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'results' not in st.session_state:
    st.session_state.results = {}

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            st.session_state.data = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! ({len(st.session_state.data)} rows)")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    if st.session_state.data is not None:
        st.subheader("Column Selection")
        columns = st.session_state.data.columns.tolist()
        
        date_column = st.selectbox("Select Date Column", columns)
        value_column = st.selectbox("Select Value Column", [col for col in columns if col != date_column])
        
        st.subheader("Forecasting Parameters")
        forecast_horizon = st.slider("Forecast Horizon (periods)", 1, 365, 30)
        test_size = st.slider("Test Size (% of data)", 10, 40, 20)
        
        st.subheader("Select Models")
        use_sarima = st.checkbox("SARIMA", value=True)
        use_holt_winters = st.checkbox("Holt-Winters", value=True)
        use_tbats = st.checkbox("TBATS", value=True)
        use_prophet = st.checkbox("Prophet", value=True)
        use_xgboost = st.checkbox("XGBoost", value=True)
        
        # Advanced parameters
        with st.expander("Advanced Settings"):
            sarima_order = st.text_input("SARIMA Order (p,d,q)", "(1,1,1)")
            sarima_seasonal = st.text_input("SARIMA Seasonal (P,D,Q,s)", "(1,1,1,12)")
            seasonality_mode = st.selectbox("Prophet Seasonality", ['additive', 'multiplicative'])
        
        run_forecast = st.button("🚀 Run Forecasting", type="primary", use_container_width=True)

# Main content
if st.session_state.data is not None:
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "📈 Forecasts", "📉 Model Comparison", "📋 Detailed Results"])
    
    with tab1:
        st.markdown('<p class="sub-header">Data Preview</p>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(st.session_state.data))
        with col2:
            st.metric("Columns", len(st.session_state.data.columns))
        with col3:
            st.metric("Date Range", f"{st.session_state.data[date_column].min()} to {st.session_state.data[date_column].max()}")
        
        st.dataframe(st.session_state.data.head(10), use_container_width=True)
        
        # Data visualization
        st.markdown('<p class="sub-header">Time Series Visualization</p>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(st.session_state.data[date_column]),
            y=st.session_state.data[value_column],
            mode='lines',
            name='Actual',
            line=dict(color='#1f77b4', width=2)
        ))
        fig.update_layout(
            title='Time Series Data',
            xaxis_title='Date',
            yaxis_title='Value',
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        st.markdown('<p class="sub-header">Descriptive Statistics</p>', unsafe_allow_html=True)
        st.dataframe(st.session_state.data[value_column].describe(), use_container_width=True)
    
    # Forecasting logic
    if run_forecast:
        with st.spinner('Running forecasting models... This may take a few minutes.'):
            try:
                # Prepare data
                df = st.session_state.data.copy()
                df[date_column] = pd.to_datetime(df[date_column])
                df = df.sort_values(date_column)
                df.set_index(date_column, inplace=True)
                
                # Train-test split
                split_idx = int(len(df) * (1 - test_size/100))
                train = df.iloc[:split_idx]
                test = df.iloc[split_idx:]
                
                results = {}
                
                # SARIMA
                if use_sarima:
                    with st.spinner('Training SARIMA...'):
                        try:
                            order = eval(sarima_order)
                            seasonal_order = eval(sarima_seasonal)
                            
                            model_sarima = SARIMAX(train[value_column], 
                                                   order=order, 
                                                   seasonal_order=seasonal_order,
                                                   enforce_stationarity=False,
                                                   enforce_invertibility=False)
                            fitted_sarima = model_sarima.fit(disp=False)
                            
                            # Predictions on test set
                            pred_test_sarima = fitted_sarima.forecast(steps=len(test))
                            
                            # Future forecast
                            pred_future_sarima = fitted_sarima.forecast(steps=forecast_horizon)
                            
                            # Calculate metrics
                            mse = mean_squared_error(test[value_column], pred_test_sarima)
                            mae = mean_absolute_error(test[value_column], pred_test_sarima)
                            mape = mean_absolute_percentage_error(test[value_column], pred_test_sarima) * 100
                            rmse = np.sqrt(mse)
                            
                            results['SARIMA'] = {
                                'test_predictions': pred_test_sarima,
                                'future_predictions': pred_future_sarima,
                                'mse': mse,
                                'mae': mae,
                                'rmse': rmse,
                                'mape': mape
                            }
                            st.success('✅ SARIMA completed')
                        except Exception as e:
                            st.warning(f'⚠️ SARIMA failed: {str(e)}')
                
                # Holt-Winters
                if use_holt_winters:
                    with st.spinner('Training Holt-Winters...'):
                        try:
                            model_hw = ExponentialSmoothing(train[value_column], 
                                                           seasonal_periods=12,
                                                           trend='add',
                                                           seasonal='add')
                            fitted_hw = model_hw.fit()
                            
                            pred_test_hw = fitted_hw.forecast(steps=len(test))
                            pred_future_hw = fitted_hw.forecast(steps=forecast_horizon)
                            
                            mse = mean_squared_error(test[value_column], pred_test_hw)
                            mae = mean_absolute_error(test[value_column], pred_test_hw)
                            mape = mean_absolute_percentage_error(test[value_column], pred_test_hw) * 100
                            rmse = np.sqrt(mse)
                            
                            results['Holt-Winters'] = {
                                'test_predictions': pred_test_hw,
                                'future_predictions': pred_future_hw,
                                'mse': mse,
                                'mae': mae,
                                'rmse': rmse,
                                'mape': mape
                            }
                            st.success('✅ Holt-Winters completed')
                        except Exception as e:
                            st.warning(f'⚠️ Holt-Winters failed: {str(e)}')
                
                # TBATS
                if use_tbats:
                    with st.spinner('Training TBATS...'):
                        try:
                            estimator = TBATS(seasonal_periods=[12])
                            fitted_tbats = estimator.fit(train[value_column].values)
                            
                            pred_test_tbats = fitted_tbats.forecast(steps=len(test))
                            pred_future_tbats = fitted_tbats.forecast(steps=forecast_horizon)
                            
                            mse = mean_squared_error(test[value_column], pred_test_tbats)
                            mae = mean_absolute_error(test[value_column], pred_test_tbats)
                            mape = mean_absolute_percentage_error(test[value_column], pred_test_tbats) * 100
                            rmse = np.sqrt(mse)
                            
                            results['TBATS'] = {
                                'test_predictions': pred_test_tbats,
                                'future_predictions': pred_future_tbats,
                                'mse': mse,
                                'mae': mae,
                                'rmse': rmse,
                                'mape': mape
                            }
                            st.success('✅ TBATS completed')
                        except Exception as e:
                            st.warning(f'⚠️ TBATS failed: {str(e)}')
                
                # Prophet
                if use_prophet:
                    with st.spinner('Training Prophet...'):
                        try:
                            prophet_train = train.reset_index().rename(columns={date_column: 'ds', value_column: 'y'})
                            
                            model_prophet = Prophet(seasonality_mode=seasonality_mode)
                            model_prophet.fit(prophet_train)
                            
                            # Test predictions
                            future_test = test.reset_index()[[date_column]].rename(columns={date_column: 'ds'})
                            pred_test_prophet = model_prophet.predict(future_test)['yhat'].values
                            
                            # Future predictions
                            future_dates = model_prophet.make_future_dataframe(periods=forecast_horizon)
                            pred_future_prophet = model_prophet.predict(future_dates)['yhat'].iloc[-forecast_horizon:].values
                            
                            mse = mean_squared_error(test[value_column], pred_test_prophet)
                            mae = mean_absolute_error(test[value_column], pred_test_prophet)
                            mape = mean_absolute_percentage_error(test[value_column], pred_test_prophet) * 100
                            rmse = np.sqrt(mse)
                            
                            results['Prophet'] = {
                                'test_predictions': pred_test_prophet,
                                'future_predictions': pred_future_prophet,
                                'mse': mse,
                                'mae': mae,
                                'rmse': rmse,
                                'mape': mape
                            }
                            st.success('✅ Prophet completed')
                        except Exception as e:
                            st.warning(f'⚠️ Prophet failed: {str(e)}')
                
                # XGBoost
                if use_xgboost:
                    with st.spinner('Training XGBoost...'):
                        try:
                            # Create lag features
                            def create_features(data, lags=12):
                                df_feat = pd.DataFrame(index=data.index)
                                for i in range(1, lags + 1):
                                    df_feat[f'lag_{i}'] = data.shift(i)
                                df_feat['rolling_mean_3'] = data.rolling(window=3).mean()
                                df_feat['rolling_std_3'] = data.rolling(window=3).std()
                                return df_feat.dropna()
                            
                            train_features = create_features(train[value_column])
                            train_target = train[value_column].loc[train_features.index]
                            
                            model_xgb = xgb.XGBRegressor(
                                n_estimators=100,
                                max_depth=5,
                                learning_rate=0.1,
                                random_state=42
                            )
                            model_xgb.fit(train_features, train_target)
                            
                            # Test predictions
                            test_features = create_features(pd.concat([train[value_column], test[value_column]]))
                            test_features = test_features.loc[test.index]
                            pred_test_xgb = model_xgb.predict(test_features)
                            
                            # Future predictions (iterative)
                            last_values = df[value_column].values.copy()
                            future_preds = []
                            
                            for _ in range(forecast_horizon):
                                features = []
                                for i in range(1, 13):
                                    features.append(last_values[-i] if len(last_values) >= i else 0)
                                features.append(np.mean(last_values[-3:]))
                                features.append(np.std(last_values[-3:]))
                                
                                pred = model_xgb.predict(np.array(features).reshape(1, -1))[0]
                                future_preds.append(pred)
                                last_values = np.append(last_values, pred)
                            
                            mse = mean_squared_error(test[value_column], pred_test_xgb)
                            mae = mean_absolute_error(test[value_column], pred_test_xgb)
                            mape = mean_absolute_percentage_error(test[value_column], pred_test_xgb) * 100
                            rmse = np.sqrt(mse)
                            
                            results['XGBoost'] = {
                                'test_predictions': pred_test_xgb,
                                'future_predictions': np.array(future_preds),
                                'mse': mse,
                                'mae': mae,
                                'rmse': rmse,
                                'mape': mape
                            }
                            st.success('✅ XGBoost completed')
                        except Exception as e:
                            st.warning(f'⚠️ XGBoost failed: {str(e)}')
                
                st.session_state.results = results
                st.session_state.train = train
                st.session_state.test = test
                st.session_state.forecast_horizon = forecast_horizon
                st.session_state.value_column = value_column
                
                st.success('🎉 All selected models completed successfully!')
                
            except Exception as e:
                st.error(f"Error during forecasting: {str(e)}")
    
    # Display results
    if st.session_state.results:
        with tab2:
            st.markdown('<p class="sub-header">Forecast Visualizations</p>', unsafe_allow_html=True)
            
            for model_name, result in st.session_state.results.items():
                st.markdown(f"### {model_name}")
                
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=st.session_state.train.index,
                    y=st.session_state.train[st.session_state.value_column],
                    mode='lines',
                    name='Training Data',
                    line=dict(color='blue', width=2)
                ))
                
                # Test data
                fig.add_trace(go.Scatter(
                    x=st.session_state.test.index,
                    y=st.session_state.test[st.session_state.value_column],
                    mode='lines',
                    name='Actual Test',
                    line=dict(color='green', width=2)
                ))
                
                # Test predictions
                fig.add_trace(go.Scatter(
                    x=st.session_state.test.index,
                    y=result['test_predictions'],
                    mode='lines',
                    name='Test Predictions',
                    line=dict(color='orange', width=2, dash='dash')
                ))
                
                # Future predictions
                last_date = st.session_state.test.index[-1]
                future_dates = pd.date_range(start=last_date, periods=st.session_state.forecast_horizon + 1, freq='D')[1:]
                
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=result['future_predictions'],
                    mode='lines',
                    name='Future Forecast',
                    line=dict(color='red', width=2, dash='dot')
                ))
                
                fig.update_layout(
                    title=f'{model_name} Forecast',
                    xaxis_title='Date',
                    yaxis_title='Value',
                    hovermode='x unified',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("RMSE", f"{result['rmse']:.2f}")
                with col2:
                    st.metric("MAE", f"{result['mae']:.2f}")
                with col3:
                    st.metric("MAPE", f"{result['mape']:.2f}%")
                with col4:
                    st.metric("MSE", f"{result['mse']:.2f}")
                
                st.markdown("---")
        
        with tab3:
            st.markdown('<p class="sub-header">Model Comparison</p>', unsafe_allow_html=True)
            
            # Create comparison DataFrame
            comparison_data = []
            for model_name, result in st.session_state.results.items():
                comparison_data.append({
                    'Model': model_name,
                    'RMSE': result['rmse'],
                    'MAE': result['mae'],
                    'MAPE': result['mape'],
                    'MSE': result['mse']
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('RMSE')
            
            # Best model
            best_model = comparison_df.iloc[0]['Model']
            st.success(f"🏆 **Recommended Model: {best_model}** (Lowest RMSE)")
            
            # Display comparison table
            st.dataframe(comparison_df.style.highlight_min(axis=0, subset=['RMSE', 'MAE', 'MAPE', 'MSE'], color='lightgreen'), use_container_width=True)
            
            # Comparison charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig_rmse = go.Figure(data=[
                    go.Bar(x=comparison_df['Model'], y=comparison_df['RMSE'], 
                          marker_color='skyblue')
                ])
                fig_rmse.update_layout(title='RMSE Comparison', xaxis_title='Model', yaxis_title='RMSE')
                st.plotly_chart(fig_rmse, use_container_width=True)
            
            with col2:
                fig_mape = go.Figure(data=[
                    go.Bar(x=comparison_df['Model'], y=comparison_df['MAPE'], 
                          marker_color='lightcoral')
                ])
                fig_mape.update_layout(title='MAPE Comparison (%)', xaxis_title='Model', yaxis_title='MAPE')
                st.plotly_chart(fig_mape, use_container_width=True)
            
            # Combined forecast comparison
            st.markdown('<p class="sub-header">Combined Forecast View</p>', unsafe_allow_html=True)
            fig_combined = go.Figure()
            
            # Actual data
            fig_combined.add_trace(go.Scatter(
                x=st.session_state.test.index,
                y=st.session_state.test[st.session_state.value_column],
                mode='lines',
                name='Actual',
                line=dict(color='black', width=3)
            ))
            
            # All model predictions
            colors = ['red', 'blue', 'green', 'orange', 'purple']
            for i, (model_name, result) in enumerate(st.session_state.results.items()):
                fig_combined.add_trace(go.Scatter(
                    x=st.session_state.test.index,
                    y=result['test_predictions'],
                    mode='lines',
                    name=model_name,
                    line=dict(color=colors[i % len(colors)], width=2, dash='dash')
                ))
            
            fig_combined.update_layout(
                title='All Models Test Predictions vs Actual',
                xaxis_title='Date',
                yaxis_title='Value',
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig_combined, use_container_width=True)
        
        with tab4:
            st.markdown('<p class="sub-header">Detailed Results</p>', unsafe_allow_html=True)
            
            for model_name, result in st.session_state.results.items():
                with st.expander(f"📊 {model_name} - Detailed Results"):
                    st.write("**Performance Metrics:**")
                    metrics_df = pd.DataFrame({
                        'Metric': ['RMSE', 'MAE', 'MAPE', 'MSE'],
                        'Value': [result['rmse'], result['mae'], f"{result['mape']:.2f}%", result['mse']]
                    })
                    st.dataframe(metrics_df, use_container_width=True)
                    
                    st.write("**Test Predictions (First 10):**")
                    test_results_df = pd.DataFrame({
                        'Date': st.session_state.test.index[:10],
                        'Actual': st.session_state.test[st.session_state.value_column].values[:10],
                        'Predicted': result['test_predictions'][:10],
                        'Error': st.session_state.test[st.session_state.value_column].values[:10] - result['test_predictions'][:10]
                    })
                    st.dataframe(test_results_df, use_container_width=True)
                    
                    st.write("**Future Forecast (First 10 periods):**")
                    last_date = st.session_state.test.index[-1]
                    future_dates = pd.date_range(start=last_date, periods=st.session_state.forecast_horizon + 1, freq='D')[1:]
                    
                    future_df = pd.DataFrame({
                        'Date': future_dates[:10],
                        'Predicted Value': result['future_predictions'][:10]
                    })
                    st.dataframe(future_df, use_container_width=True)
                    
                    # Download predictions
                    full_future_df = pd.DataFrame({
                        'Date': future_dates,
                        'Predicted_Value': result['future_predictions']
                    })
                    csv = full_future_df.to_csv(index=False)
                    st.download_button(
                        label=f"📥 Download {model_name} Forecast",
                        data=csv,
                        file_name=f"{model_name}_forecast.csv",
                        mime="text/csv"
                    )

else:
    st.info("👈 Please upload a CSV file from the sidebar to begin forecasting.")
    st.markdown("""
    ### 📋 Instructions:
    1. **Upload CSV**: Click 'Browse files' in the sidebar
    2. **Select Columns**: Choose your date and value columns
    3. **Configure**: Set forecast horizon and test size
    4. **Choose Models**: Select which forecasting methods to use
    5. **Run**: Click 'Run Forecasting' button
    
    ### 📊 Available Models:
    - **SARIMA**: Seasonal AutoRegressive Integrated Moving Average
    - **Holt-Winters**: Exponential Smoothing with trend and seasonality
    - **TBATS**: Trigonometric seasonality, Box-Cox transformation, ARMA errors, Trend and Seasonal components
    - **Prophet**: Facebook's forecasting tool for time series with strong seasonal patterns
    - **XGBoost**: Gradient boosting with engineered lag features
    
    ### 📈 Features:
    - Train-test validation with customizable split
    - Multiple accuracy metrics (RMSE, MAE, MAPE, MSE)
    - Automatic best model recommendation
    - Interactive visualizations
    - Downloadable forecasts
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Advanced Time Series Forecasting Engine")
