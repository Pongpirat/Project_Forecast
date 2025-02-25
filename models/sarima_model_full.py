import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
import streamlit as st

@st.cache_data
def sarima_full_model(data, value_column, forecast_horizon, smooth_window=3, seasonal_order=None, order=None):
    """
    ใช้ข้อมูล Historical ทั้งหมด (ไม่รวมไฟล์ 30 วัน) ฝึกโมเดล
    แล้วทำนาย forecast_horizon วันข้างหน้า
    """
    try:
        train_data = data.copy()
        train_data['smoothed'] = train_data[value_column].rolling(window=smooth_window).mean()
        train_data['lag_1'] = train_data['smoothed'].shift(1)
        train_data['lag_7'] = train_data['smoothed'].shift(7)
        train_data = train_data.dropna()

        if order is None or seasonal_order is None:
            auto_model = auto_arima(
                train_data['smoothed'], 
                seasonal=True, 
                m=7,  
                trace=False,
                error_action='ignore',
                suppress_warnings=True
            )
            order = auto_model.order
            seasonal_order = auto_model.seasonal_order

        sarima_model_obj = SARIMAX(
            train_data['smoothed'],
            order=order,
            seasonal_order=seasonal_order
        )
        sarima_model_fit = sarima_model_obj.fit(disp=False)

        forecast = sarima_model_fit.get_forecast(steps=forecast_horizon).predicted_mean

        last_date = data.index[-1]
        forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                       periods=forecast_horizon,
                                       freq='D')
        forecast_series = pd.Series(forecast, index=forecast_index)

        # Prepend ค่าจริงวันล่าสุด
        last_actual = pd.Series([data[value_column].iloc[-1]], index=[last_date])
        combined = pd.concat([last_actual, forecast_series])

        return {
            'comparison': combined.to_frame(name='Predicted'),
            'mae': None,
            'rmse': None,
            'mape': None,
            'order': order,
            'seasonal_order': seasonal_order
        }
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในโมเดล SARIMA (Full Data): {e}")
        raise e
