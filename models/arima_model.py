import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st

def arima_model(data, value_column, days_to_remove, order=(1, 1, 1)):
    try:
        # แยกข้อมูล Train และ Test
        series_data = data[value_column]
        train_data = series_data[:-days_to_remove]
        test_data = series_data[-days_to_remove:]

        # สร้างโมเดล ARIMA
        model = ARIMA(train_data, order=order)
        model_fit = model.fit()

        # ทำนายค่า
        forecast = model_fit.forecast(steps=len(test_data))

        # สร้าง DataFrame สำหรับการเปรียบเทียบ
        comparison = pd.DataFrame({
            'Actual': series_data,
            'Predicted': np.nan
        })
        comparison.loc[test_data.index, 'Predicted'] = forecast.values

        # คำนวณ Metrics
        mae = mean_absolute_error(test_data, forecast)
        rmse = np.sqrt(mean_squared_error(test_data, forecast))
        mape = np.mean(np.abs((test_data - forecast) / test_data)) * 100

        # ส่งผลลัพธ์กลับ
        return {
            'comparison': comparison,
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในโมเดล ARIMA: {e}")
        raise e