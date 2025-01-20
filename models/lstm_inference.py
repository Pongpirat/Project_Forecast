import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

def inference_lstm_model(data, value_column, days_to_remove, model_path, scaler_path, window_size):
    # ---------- 1) โหลด Scaler ----------
    scaler = joblib.load(scaler_path)

    # ---------- 2) สเกลข้อมูล ----------
    scaled_data = scaler.transform(data)

    # ---------- 3) แบ่ง Train/Test ----------
    test_scaled = scaled_data[-days_to_remove:]

    # ---------- 4) โหลดโมเดล LSTM ----------
    model = load_model(model_path)

    # ---------- 5) เตรียม Sequence สำหรับการทำนาย ----------
    last_train_sequence = scaled_data[-(window_size + days_to_remove):-days_to_remove]

    predictions = []

    for i in range(days_to_remove):
        # สร้าง Input Sequence สำหรับโมเดล
        input_seq = last_train_sequence.reshape(1, window_size, last_train_sequence.shape[1])
        
        # พยากรณ์ค่าถัดไป
        pred = model.predict(input_seq)[0][0]
        predictions.append(pred)

        # อัปเดต Sequence ใหม่
        new_row = test_scaled[i].reshape(1, -1)  # แถวถัดไปจาก Test Set
        new_row[0, list(data.columns).index(value_column)] = pred  # แทนค่าที่พยากรณ์ลงใน value_column

        # อัปเดต Sequence เพื่อใช้พยากรณ์ต่อ
        last_train_sequence = np.append(last_train_sequence[1:], new_row, axis=0)

    # ---------- 6) Inverse Transform ค่าที่พยากรณ์และจริง ----------
    predictions_full = np.zeros((days_to_remove, scaled_data.shape[1]))
    predictions_full[:, list(data.columns).index(value_column)] = predictions

    actual_full = np.zeros((days_to_remove, scaled_data.shape[1]))
    actual_full[:, list(data.columns).index(value_column)] = test_scaled[:, list(data.columns).index(value_column)]

    predictions_inversed = scaler.inverse_transform(predictions_full)[:, list(data.columns).index(value_column)]
    actual_inversed = scaler.inverse_transform(actual_full)[:, list(data.columns).index(value_column)]

    # ---------- 7) สร้าง DataFrame เปรียบเทียบ ----------
    comparison = pd.DataFrame({
        'Actual': actual_inversed,
        'Predicted': predictions_inversed
    }, index=data.index[-days_to_remove:])

    # ---------- 8) คำนวณ Metrics ----------
    mae = np.mean(np.abs(actual_inversed - predictions_inversed))
    rmse = np.sqrt(np.mean((actual_inversed - predictions_inversed)**2))
    mape = np.mean(np.abs((actual_inversed - predictions_inversed) / actual_inversed)) * 100

    return {
        'comparison': comparison,
        'mae': mae,
        'rmse': rmse,
        'mape': mape
    } 