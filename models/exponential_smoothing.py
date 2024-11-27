# models/exponential_smoothing.py

import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import streamlit as st

def exponential_smoothing_model(data, value_column, days_to_remove):
    try:
        # เตรียมข้อมูล
        series_data = data[value_column]
        train_data = series_data[:-days_to_remove]
        test_data = series_data[-days_to_remove:]

        # กำหนดพารามิเตอร์ของโมเดล
        trend_option = 'mul'
        seasonal_option = 'mul'
        seasonal_periods_option = 365

        # ฝึกโมเดล
        with st.spinner('กำลังฝึกโมเดล Exponential Smoothing...'):
            model = ExponentialSmoothing(
                train_data,
                trend=trend_option,
                seasonal=seasonal_option,
                seasonal_periods=seasonal_periods_option,
                initialization_method="estimated"
            )
            fitted_model = model.fit(optimized=True)

        # ทำนาย
        with st.spinner('กำลังทำนายและวิเคราะห์ผลลัพธ์...'):
            forecast = fitted_model.forecast(steps=days_to_remove)

            # สร้าง DataFrame สำหรับการเปรียบเทียบ
            comparison = pd.DataFrame({
                'Actual': series_data,
                'Predicted': np.nan
            })
            comparison.loc[test_data.index, 'Predicted'] = forecast

            # คำนวณค่า Metric สำหรับช่วงทดสอบ
            mae = np.mean(np.abs(test_data - forecast))
            rmse = np.sqrt(np.mean((test_data - forecast)**2))
            mape = np.mean(np.abs((test_data - forecast) / test_data)) * 100

        # ส่งผลลัพธ์กลับ
        return {
            'comparison': comparison,
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในโมเดล Exponential Smoothing: {e}")
        raise e