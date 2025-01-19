import numpy as np
import pandas as pd
import os
import joblib

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from data_processing import process_data  # ฟังก์ชันเตรียมข้อมูลที่คุณมี

def main():
    # ---------- 1) กำหนด path ของไฟล์ CSV ----------
    csv_file_path = os.path.join('currency', 'AUD.csv')  # แก้ไขชื่อไฟล์ตามจริง

    # ---------- 2) โหลดและตรวจสอบข้อมูล ----------
    try:
        df = pd.read_csv(csv_file_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_file_path, encoding='cp874')
        except Exception as e:
            print(f"ไม่สามารถอ่านไฟล์ได้: {e}")
            return

    if 'งวด' not in df.columns:
        print("ไม่พบคอลัมน์ 'งวด' ในไฟล์ CSV ที่เลือก")
        return

    # ---------- 3) ประมวลผลข้อมูล (เติมวัน, forward-fill ฯลฯ) ----------
    df_processed, numeric_cols = process_data(df)

    value_column = 'อัตราขาย'  # คอลัมน์ที่ต้องการพยากรณ์
    if value_column not in numeric_cols:
        print(f"ไม่พบคอลัมน์ '{value_column}' ในข้อมูลที่ผ่านการประมวลผล")
        return

    # ---------- 4) สร้างข้อมูลสำหรับ LSTM ----------
    series_data = df_processed[value_column].values.reshape(-1, 1)

    # (4.1) สเกลข้อมูลด้วย MinMaxScaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(series_data)

    # ---------- 5) บันทึก scaler ไว้ใช้ตอน Inference ----------
    scaler_path = 'models/scaler_lstm.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"Scaler ถูกบันทึกไว้ที่ {scaler_path}")

    # ---------- 6) กำหนดพารามิเตอร์สำหรับการพยากรณ์ ----------
    window_size = 30    # จำนวนวันย้อนหลังที่ให้โมเดลมอง
    days_to_remove = 30 # จำนวนวันท้ายสุดที่กันไว้เป็น Test Set

    # ---------- 7) แบ่ง Train/Test ----------
    train_data = scaled_data[:-days_to_remove]
    test_data = scaled_data[-days_to_remove:]

    # ---------- 8) สร้าง sequences สำหรับ Train ----------
    X_train, y_train = [], []
    for i in range(window_size, len(train_data)):
        X_train.append(train_data[i-window_size:i])
        y_train.append(train_data[i])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # ---------- 9) แบ่ง Train/Validation (เช่น 80/20) ----------
    train_size = int(len(X_train) * 0.8)
    X_val = X_train[train_size:]
    y_val = y_train[train_size:]
    X_train2 = X_train[:train_size]
    y_train2 = y_train[:train_size]

    # ---------- 10) สร้างโมเดล LSTM (Multi-Layer + Dropout + Learning Rate) ----------
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(window_size, 1)),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])

    # ปรับ Optimizer เป็น Adam + ลด Learning Rate
    optimizer = Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # ---------- 11) ใช้ EarlyStopping ป้องกัน Overfitting ----------
    early_stop = EarlyStopping(
        monitor='val_loss',  # ตรวจสอบ loss บน validation
        patience=10,         # หาก val_loss ไม่ดีขึ้นเลย 10 epoch ให้หยุด
        restore_best_weights=True  # ดึง weights ที่ดีที่สุดกลับมา
    )

    epochs = 200
    batch_size = 16

    print("กำลังฝึกโมเดล LSTM (แบบปรับปรุง)...")
    model.fit(
        X_train2, y_train2,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )
    print("การฝึกโมเดลเสร็จสิ้นแล้ว")

    # ---------- 12) บันทึกโมเดล ----------
    model_path = 'models/lstm_model.h5'
    model.save(model_path)
    print(f"โมเดลถูกบันทึกไว้ที่ {model_path}")

    # ---------- 13) ประเมินโมเดลบน Test Set (Walk-Forward) ----------
    last_train_sequence = train_data[-window_size:]  # (window_size, 1)
    predictions = []
    print("กำลังทำนายข้อมูลบน Test Set...")

    for i in range(days_to_remove):
        input_seq = last_train_sequence.reshape(1, window_size, 1)
        pred = model.predict(input_seq)
        predictions.append(pred[0][0])
        # อัปเดต sequence
        last_train_sequence = np.append(last_train_sequence[1:], pred, axis=0)

    # ---------- 14) Inverse Transform ----------
    predictions_inversed = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).ravel()
    actual_inversed = scaler.inverse_transform(test_data).ravel()

    # ---------- 15) สร้าง DataFrame เปรียบเทียบ ----------
    comparison = pd.DataFrame({
        'Actual': actual_inversed,
        'Predicted': predictions_inversed
    }, index=df_processed.index[-days_to_remove:])

    # ---------- 16) คำนวณ Metrics ----------
    mae = np.mean(np.abs(actual_inversed - predictions_inversed))
    rmse = np.sqrt(np.mean((actual_inversed - predictions_inversed)**2))
    mape = np.mean(np.abs((actual_inversed - predictions_inversed) / actual_inversed)) * 100

    print("ผลการประเมินโมเดล (Improved) บน Test Set:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Square Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

    # ---------- 17) บันทึก Metrics & Comparison (option) ----------
    metrics_path = 'models/lstm_metrics.csv'
    comparison_path = 'models/lstm_comparison.csv'

    pd.DataFrame([{'MAE': mae, 'RMSE': rmse, 'MAPE': mape}]).to_csv(metrics_path, index=False)
    comparison.to_csv(comparison_path)
    print(f"Metrics ถูกบันทึกไว้ที่ {metrics_path}")
    print(f"Comparison ถูกบันทึกไว้ที่ {comparison_path}")

if __name__ == "__main__":
    main()
