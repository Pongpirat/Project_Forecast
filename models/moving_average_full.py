import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data
def moving_average_full_model(data, value_column, forecast_horizon, window_size=7, use_ema=True):
    """
    ใช้ข้อมูล Historical ทั้งหมด (ไม่รวมไฟล์ 30 วัน) ฝึกโมเดล
    แล้วทำนาย forecast_horizon วันข้างหน้า
    """
    try:
        series_data = data[value_column]
        train_data = series_data

        forecast_values = []
        rolling_data = train_data.copy()

        for i in range(forecast_horizon):
            if len(rolling_data) >= window_size:
                window_data = rolling_data.iloc[-window_size:]
            else:
                window_data = rolling_data

            if use_ema:
                moving_avg = window_data.ewm(span=window_size, adjust=False).mean().iloc[-1]
            else:
                moving_avg = window_data.mean()

            forecast_values.append(moving_avg)
            rolling_data = pd.concat([rolling_data, pd.Series([moving_avg])])

        last_date = series_data.index[-1]
        forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                       periods=forecast_horizon,
                                       freq='D')
        forecast_series = pd.Series(forecast_values, index=forecast_index)

        # Prepend ค่าจริงวันล่าสุด
        last_actual = pd.Series([series_data.iloc[-1]], index=[last_date])
        combined = pd.concat([last_actual, forecast_series])

        return {
            'comparison': combined.to_frame(name='Predicted'),
            'mae': None,
            'rmse': None,
            'mape': None
        }
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในโมเดล Moving Average (Full Data): {e}")
        raise e
