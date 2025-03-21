# lstm_inference_full.py
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

def inference_lstm_full_model(
    data,
    value_column,
    model_path,
    scaler_date_path,
    scaler_target_path,
    window_size,
    forecast_days=365
):
    """
    Inference สำหรับโมเดล LSTM (Full Data) เพื่อทำนาย 30 วันข้างหน้า
    โดยใช้โมเดลและ Scaler ที่ฝึกจาก train_lstm_full.py
    และมีการจำกัดการเพิ่มขึ้นของค่าพยากรณ์ (clamping) ไม่ให้เกิน 5% ต่อวัน
    """
    # ตรวจสอบให้แน่ใจว่ามี lag features อยู่ในข้อมูล
    if 'lag_1' not in data.columns:
        data['lag_1'] = data[value_column].shift(1)
    if 'lag_2' not in data.columns:
        data['lag_2'] = data[value_column].shift(2)
    data.dropna(inplace=True)

    date_features = ['day', 'month', 'year', 'week']
    target_features = [value_column, 'lag_1', 'lag_2']
    all_features = date_features + target_features

    scaler_date = joblib.load(scaler_date_path)
    scaler_target = joblib.load(scaler_target_path)
    model = load_model(model_path)

    # Scale ข้อมูลทั้งหมด
    full_date_scaled = scaler_date.transform(data[date_features])
    full_target_scaled = scaler_target.transform(data[target_features])
    full_scaled = np.hstack([full_date_scaled, full_target_scaled])
    full_scaled_df = pd.DataFrame(full_scaled, index=data.index, columns=all_features)

    # เตรียม sequence สุดท้าย (window_size แถวสุดท้าย) สำหรับทำนาย
    last_sequence = full_scaled_df.iloc[-window_size:].copy()

    # สร้าง future_dates โดยเริ่มจากวันถัดไปของข้อมูลล่าสุด
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                 periods=forecast_days,
                                 freq='D')
    future_features_date = pd.DataFrame(index=future_dates)
    future_features_date['day']   = future_features_date.index.day
    future_features_date['month'] = future_features_date.index.month
    future_features_date['year']  = future_features_date.index.year
    future_features_date['week']  = future_features_date.index.isocalendar().week

    future_pred_list = []
    idx_target = len(date_features)
    
    # กำหนดอัตราสูงสุดที่อนุญาตให้เพิ่มขึ้นต่อวัน (เช่น 5% ต่อวัน)
    max_increase_ratio = 1.20

    # ทำ Recursive Forecast สำหรับ forecast_days วัน
    for i in range(forecast_days):
        # Scale ข้อมูลสำหรับวันอนาคต
        row_date_unscaled = future_features_date.iloc[i]
        row_date_scaled = scaler_date.transform([row_date_unscaled.values])
        
        # สร้าง input sequence สำหรับโมเดล (reshape เป็น (1, window_size, n_features))
        input_X = last_sequence.values.reshape(1, window_size, len(all_features))
        pred_scaled_value = model.predict(input_X)[0][0]
        
        # อัปเดตค่า lag สำหรับ sequence ใหม่
        prev_lag_1_s = last_sequence.iloc[-1, idx_target + 1]  # lag_1 ของแถวสุดท้ายใน sequence
        new_lag_1_s = pred_scaled_value
        new_lag_2_s = prev_lag_1_s
        
        row_target_scaled = np.array([[pred_scaled_value, new_lag_1_s, new_lag_2_s]])
        new_scaled_row = np.hstack([row_date_scaled, row_target_scaled])
        
        new_sequence = np.vstack([last_sequence.values[1:], new_scaled_row])
        last_sequence = pd.DataFrame(new_sequence, columns=all_features)
        
        # ทำ inverse scaling เฉพาะส่วน target (อัตราขาย)
        tmp_full = np.zeros((1, len(all_features)))
        tmp_full[0, idx_target] = pred_scaled_value
        tmp_target_part = tmp_full[:, len(date_features):]
        pred_inv_value = scaler_target.inverse_transform(tmp_target_part)[0, 0]
        
        # *** Clamping ***: จำกัดไม่ให้ค่าพยากรณ์เพิ่มขึ้นเกิน 5% เมื่อเทียบกับวันก่อนหน้า
        if i == 0:
            base_value = data[value_column].iloc[-1]
        else:
            base_value = future_pred_list[-1]
        max_allowed = base_value * max_increase_ratio
        if pred_inv_value > max_allowed:
            pred_inv_value = max_allowed
        
        future_pred_list.append(pred_inv_value)

    forecast_df = pd.DataFrame({'Predicted': future_pred_list}, index=future_dates)

    # Prepend ค่า actual ของวันล่าสุดลงใน forecast_df เพื่อให้กราฟต่อเนื่อง
    last_actual = pd.DataFrame({'Predicted': [data[value_column].iloc[-1]]}, index=[last_date])
    forecast_df = pd.concat([last_actual, forecast_df])

    return forecast_df
