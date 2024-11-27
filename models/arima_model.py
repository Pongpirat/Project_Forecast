import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st

def arima_model(data, value_column, days_to_remove, order=(1, 1, 1)):
    try:
        # เตรียมข้อมูล
        series_data = data[value_column]
        train_data = series_data[:-days_to_remove]
        test_data = series_data[-days_to_remove:]
        
        # สร้างและฝึกโมเดล ARIMA
        with st.spinner('กำลังฝึกโมเดล ARIMA...'):
            model = ARIMA(train_data, order=order)
            model_fit = model.fit()
        
        # พยากรณ์ข้อมูล
        with st.spinner('กำลังพยากรณ์ข้อมูลด้วย ARIMA...'):
            forecast = model_fit.forecast(steps=days_to_remove)
        
        # สร้าง DataFrame สำหรับการเปรียบเทียบ
        comparison = pd.DataFrame({
            'Actual': series_data,
            'Predicted': np.nan
        })
        comparison.loc[test_data.index, 'Predicted'] = forecast
        
        # คำนวณค่า Metric สำหรับช่วงทดสอบ
        mae = mean_absolute_error(test_data, forecast)
        rmse = np.sqrt(mean_squared_error(test_data, forecast))
        mape = np.mean(np.abs((test_data - forecast) / test_data)) * 100
        
        # สร้างกราฟ
        fig_original = go.Figure()
        fig_original.add_trace(go.Scatter(
            x=comparison.index,
            y=comparison['Actual'],
            mode='lines',
            name='Original Data'
        ))
        fig_original.update_layout(
            title="Original Data",
            xaxis_title="Date",
            yaxis_title=value_column,
            template="plotly_white"
        )
        
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(
            x=comparison.index,
            y=comparison['Predicted'],
            mode='lines',
            name='Forecasted Data'
        ))
        fig_forecast.update_layout(
            title="Forecasted Data",
            xaxis_title="Date",
            yaxis_title=value_column,
            template="plotly_white"
        )
        
        fig_combined = go.Figure()
        fig_combined.add_trace(go.Scatter(
            x=comparison.index,
            y=comparison['Actual'],
            mode='lines',
            name='Original Data'
        ))
        fig_combined.add_trace(go.Scatter(
            x=comparison.index,
            y=comparison['Predicted'],
            mode='lines',
            name='Forecasted Data'
        ))
        fig_combined.update_layout(
            title="Original vs Forecasted Data",
            xaxis_title="Date",
            yaxis_title=value_column,
            template="plotly_white"
        )
        
        # ส่งผลลัพธ์กลับ
        return {
            'comparison': comparison,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'fig_original': fig_original,
            'fig_forecast': fig_forecast,
            'fig_combined': fig_combined
        }
    
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในโมเดล ARIMA: {e}")
        raise e