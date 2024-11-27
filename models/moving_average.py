import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st

def moving_average_model(data, value_column, days_to_remove, window_size=7, use_ema=True):
    try:
        # เตรียมข้อมูล
        series_data = data[value_column]
        train_data = series_data[:-days_to_remove]
        test_data = series_data[-days_to_remove:]

        # สร้างซีรีส์สำหรับเก็บค่าพยากรณ์
        forecast_series = pd.Series(index=test_data.index)

        # เริ่มต้นจาก train_data เท่านั้น
        rolling_data = train_data.copy()

        # ทำการพยากรณ์แบบ Recursive Rolling Forecast
        for i in range(len(test_data)):
            # ใช้ rolling_data ย้อนหลัง window_size วัน
            if len(rolling_data) >= window_size:
                window_data = rolling_data.iloc[-window_size:]
            else:
                window_data = rolling_data

            # คำนวณค่าเฉลี่ยแบบ EMA
            if use_ema:
                moving_avg = window_data.ewm(span=window_size, adjust=False).mean().iloc[-1]
            else:
                moving_avg = window_data.mean()

            # เก็บค่าพยากรณ์
            forecast_series.iloc[i] = moving_avg

            # เพิ่มค่าพยากรณ์ใหม่เข้าไปใน rolling_data ด้วย concat
            rolling_data = pd.concat([rolling_data, pd.Series([moving_avg], index=[test_data.index[i]])])

            # Debug: แสดงข้อมูลระหว่างการพยากรณ์ (สามารถคอมเมนต์ออกได้ในโปรดักชัน)
            # st.write(f"Day {i+1}: Moving Avg = {moving_avg}")

        # สร้าง DataFrame สำหรับการเปรียบเทียบ
        comparison = pd.DataFrame({
            'Actual': series_data,
            'Predicted': np.nan
        })
        comparison.loc[test_data.index, 'Predicted'] = forecast_series.values

        # คำนวณ Metrics
        mae = mean_absolute_error(test_data, forecast_series)
        rmse = np.sqrt(mean_squared_error(test_data, forecast_series))
        mape = np.mean(np.abs((test_data - forecast_series) / test_data)) * 100

        # ส่งผลลัพธ์กลับ
        return {
            'comparison': comparison,
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในโมเดล Moving Average: {e}")
        raise e