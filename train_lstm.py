import numpy as np
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from data_processing import process_data

def main():
    # ---------- 1) กำหนด path ของไฟล์ CSV ----------
    csv_file_path = os.path.join('currency', 'AUD.csv')  # ปรับชื่อไฟล์ตามจริง

    # ---------- 2) โหลดไฟล์ CSV ----------
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

    # ---------- 3) ประมวลผลข้อมูล ----------
    value_column = 'อัตราขาย'  # คอลัมน์เป้าหมาย
    df_processed, numeric_cols = process_data(df, value_column)

    if value_column not in numeric_cols:
        print(f"ไม่พบคอลัมน์ '{value_column}' ในข้อมูลที่ผ่านการประมวลผล")
        return

    # ---------- 4) สเกลข้อมูลทั้งหมด (รวม smooth_7, smooth_14) ----------
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_processed[numeric_cols])
    scaled_df = pd.DataFrame(scaled_data, index=df_processed.index, columns=numeric_cols)

    # ---------- 5) บันทึก Scaler สำหรับ Inference ----------
    scaler_path = 'models/scaler_lstm.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"Scaler ถูกบันทึกที่ {scaler_path}")

    # ---------- 6) กำหนดพารามิเตอร์ ----------
    window_size = 30
    days_to_remove = 30

    # ---------- 7) แยก Train/Test ----------
    train_scaled = scaled_df.iloc[:-days_to_remove]
    test_scaled = scaled_df.iloc[-days_to_remove:]

    # ---------- 8) สร้าง Sequence สำหรับ Train ----------
    X_train, y_train = [], []
    for i in range(window_size, len(train_scaled)):
        # ดึงฟีเจอร์ทั้งหมด (รวม smooth_x)
        X_train.append(train_scaled.iloc[i-window_size:i].values)
        # เลือก Target จากคอลัมน์หลัก (value_column)
        y_train.append(train_scaled.iloc[i][value_column])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # ---------- 9) แบ่ง Train/Validation (80:20) ----------
    train_size = int(len(X_train) * 0.8)
    X_val = X_train[train_size:]
    y_val = y_train[train_size:]
    X_train2 = X_train[:train_size]
    y_train2 = y_train[:train_size]

    # ---------- 10) สร้างโมเดล LSTM ----------
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=(window_size, X_train.shape[2])),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(1)
    ])

    optimizer = Adam(learning_rate=1e-5)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )

    epochs = 300
    batch_size = 32

    print("กำลังฝึกโมเดล LSTM...")
    history = model.fit(
        X_train2, y_train2,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )
    print("การฝึกโมเดลเสร็จสิ้นแล้ว")

    # ---------- แสดงกราฟ Loss ----------
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('models/loss_plot.png')
    plt.close()
    print("Loss plot ถูกบันทึกที่ models/loss_plot.png")

    # ---------- 11) บันทึกโมเดล ----------
    model_path = 'models/lstm_model.h5'
    model.save(model_path)
    print(f"โมเดลถูกบันทึกที่ {model_path}")

    # ---------- 12) ประเมินโมเดลบน Test Set (Simple) ----------
    # ตัวอย่างง่าย ๆ ในที่นี้ จะสร้าง Sequence จาก train_scaled ชุดสุดท้าย
    # แล้วทำนายแบบ walk-forward หรือ จะทำนายครั้งเดียวก็ได้

    # สร้าง Sequence สุดท้ายสำหรับทำนาย
    X_test, y_test = [], []
    for i in range(window_size, len(scaled_df)):
        X_test.append(scaled_df.iloc[i-window_size:i].values)
        y_test.append(scaled_df.iloc[i][value_column])
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # สร้าง test_set เฉพาะส่วน days_to_remove
    X_test_set = X_test[-days_to_remove:]
    y_test_set = y_test[-days_to_remove:]
    dates_test_set = df_processed.index[-days_to_remove:]

    # ทำนาย
    predictions_scaled = model.predict(X_test_set)
    predictions_scaled = predictions_scaled.ravel()

    # inverse transform เฉพาะคอลัมน์เป้าหมาย
    predictions_full = np.zeros((days_to_remove, len(numeric_cols)))
    predictions_full[:, numeric_cols.get_loc(value_column)] = predictions_scaled

    y_test_full = np.zeros((days_to_remove, len(numeric_cols)))
    y_test_full[:, numeric_cols.get_loc(value_column)] = y_test_set

    predictions_inversed = scaler.inverse_transform(predictions_full)[:, numeric_cols.get_loc(value_column)]
    actual_inversed = scaler.inverse_transform(y_test_full)[:, numeric_cols.get_loc(value_column)]

    # ---------- 13) สร้าง DataFrame เปรียบเทียบ ----------
    comparison = pd.DataFrame({
        'Actual': actual_inversed,
        'Predicted': predictions_inversed
    }, index=dates_test_set)

    mae = np.mean(np.abs(actual_inversed - predictions_inversed))
    rmse = np.sqrt(np.mean((actual_inversed - predictions_inversed)**2))
    mape = np.mean(np.abs((actual_inversed - predictions_inversed)/actual_inversed))*100

    print("ผลการประเมินโมเดลบน Test Set:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")

    comparison_path = 'models/lstm_comparison.csv'
    comparison.to_csv(comparison_path)
    print(f"Comparison ถูกบันทึกที่ {comparison_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(comparison['Actual'], label='Actual')
    plt.plot(comparison['Predicted'], label='Predicted')
    plt.title('Actual vs Predicted')
    plt.xlabel('Date')
    plt.ylabel(value_column)
    plt.legend()
    plt.savefig('models/actual_vs_predicted.png')
    plt.close()
    print("Actual vs Predicted plot ถูกบันทึกที่ models/actual_vs_predicted.png")

if __name__ == "__main__":
    main()