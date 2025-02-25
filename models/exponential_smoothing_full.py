import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import streamlit as st

@st.cache_data
def exponential_smoothing_full_model(data, value_column, forecast_horizon):
    """
    ใช้ข้อมูล Historical ทั้งหมด (ไม่รวมไฟล์ 30 วัน) ฝึกโมเดล
    แล้วทำนาย forecast_horizon วันข้างหน้า
    """
    try:
        series_data = data[value_column]
        train_data = series_data

        trend_option = 'mul'
        seasonal_option = 'mul'
        seasonal_periods_option = 365

        with st.spinner('ฝึกโมเดล Exponential Smoothing (Full Data)...'):
            model = ExponentialSmoothing(
                train_data,
                trend=trend_option,
                seasonal=seasonal_option,
                seasonal_periods=seasonal_periods_option,
                initialization_method="estimated"
            )
            fitted_model = model.fit(optimized=True)

        with st.spinner('ทำนาย (Full Data) ...'):
            forecast = fitted_model.forecast(steps=forecast_horizon)

            last_date = series_data.index[-1]
            forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                           periods=forecast_horizon,
                                           freq='D')
            forecast_series = pd.Series(forecast, index=forecast_index)

            # Prepend ค่าจริงวันล่าสุดเพื่อให้กราฟต่อเนื่อง
            last_actual = pd.Series([series_data.iloc[-1]], index=[last_date])
            combined = pd.concat([last_actual, forecast_series])

        return {
            'comparison': combined.to_frame(name='Predicted'),
            'mae': None,
            'rmse': None,
            'mape': None
        }
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในโมเดล Exponential Smoothing (Full Data): {e}")
        raise e
