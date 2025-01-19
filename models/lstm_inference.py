# models/lstm_inference.py

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model
import joblib
import streamlit as st

def inference_lstm_model(
    data,           # DataFrame ที่ผ่านการ process_data (และเติมวันที่) แล้ว
    value_column,   # ชื่อคอลัมน์ที่ต้องการทำนาย
    days_to_remove, # จำนวนวันที่จะกันไว้ทดสอบ
    model_path,     # path ของไฟล์ .h5 ที่ฝึกมาแล้ว
    scaler_path,    # path ของไฟล์ scaler .pkl
    window_size=30  # ต้องตรงกับ window_size ที่ใช้ตอนฝึก
):
    try:
        # 1) โหลดโมเดลและ scaler
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)

        # 2) เตรียมข้อมูล
        series_data = data[value_column].values.reshape(-1, 1)
        scaled_data = scaler.transform(series_data)

        # แยก train / test (สำหรับแสดงผลเปรียบเทียบ)
        train_data = scaled_data[:-days_to_remove]
        test_data = scaled_data[-days_to_remove:]

        # 3) สร้าง sequence สุดท้ายจาก Train เพื่อใช้ทำนายแบบ walk-forward
        #    (เอาตัวท้ายของ train_data = window_size จุด)
        last_sequence = train_data[-window_size:]  # shape (window_size, 1)

        predictions = []
        for i in range(days_to_remove):
            # reshape เป็น (1, window_size, 1)
            input_seq = last_sequence.reshape(1, window_size, 1)
            pred = model.predict(input_seq)
            predictions.append(pred[0][0])
            # อัปเดต sequence ด้วยค่าที่พยากรณ์
            last_sequence = np.append(last_sequence[1:], pred, axis=0)

        # แปลงค่าพยากรณ์กลับไปเป็นสเกลเดิม
        predictions_inversed = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).ravel()
        actual_inversed = scaler.inverse_transform(test_data).ravel()

        # 4) สร้าง DataFrame เปรียบเทียบ
        comparison = pd.DataFrame({
            "Actual": actual_inversed,
            "Predicted": predictions_inversed
        }, index=data.index[-days_to_remove:])

        # 5) คำนวณ Metrics
        mae = mean_absolute_error(actual_inversed, predictions_inversed)
        rmse = np.sqrt(mean_squared_error(actual_inversed, predictions_inversed))
        mape = np.mean(np.abs((actual_inversed - predictions_inversed) / actual_inversed)) * 100

        return {
            "comparison": comparison,
            "mae": mae,
            "rmse": rmse,
            "mape": mape
        }

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดขณะทำ Inference LSTM: {e}")
        raise e