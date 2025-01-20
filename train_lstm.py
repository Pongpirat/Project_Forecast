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
    # (1) กำหนด path ของไฟล์ CSV
    csv_file_path = os.path.join('currency', 'AUD.csv')

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

    # (2) ประมวลผลข้อมูล
    value_column = 'อัตราขาย'
    df_processed, numeric_cols = process_data(df, value_column)

    # เลือกฟีเจอร์ที่ต้องใช้ [value_column, diff_1, diff_7]
    df_processed = df_processed[[value_column, 'diff_1', 'diff_7']]

    # (3) สเกลข้อมูล
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_processed)
    scaled_df = pd.DataFrame(scaled_data, index=df_processed.index, columns=df_processed.columns)

    scaler_path = 'models/scaler_lstm.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"Scaler บันทึกไว้ที่ {scaler_path}")

    # (4) กำหนดพารามิเตอร์
    window_size = 14
    days_to_remove = 30

    # (5) แบ่ง Train/Test
    train_scaled = scaled_df.iloc[:-days_to_remove]
    test_scaled = scaled_df.iloc[-days_to_remove:]

    # (6) สร้าง Sequence สำหรับ Train
    X_train, y_train = [], []
    for i in range(window_size, len(train_scaled)):
        X_train.append(train_scaled.iloc[i-window_size:i].values)
        y_train.append(train_scaled.iloc[i][value_column])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # (7) แบ่ง Train/Validation
    train_size = int(len(X_train) * 0.8)
    X_val, y_val = X_train[train_size:], y_train[train_size:]
    X_train2, y_train2 = X_train[:train_size], y_train[:train_size]

    # (8) สร้างโมเดล LSTM
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=(window_size, X_train.shape[2])),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(1)
    ])

    optimizer = Adam(learning_rate=1e-5)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    epochs = 200
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
    print("ฝึกโมเดลเสร็จสิ้น")

    # (9) แสดงกราฟ Loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('models/loss_plot.png')
    plt.close()

    model_path = 'models/lstm_model.h5'
    model.save(model_path)
    print(f"โมเดลบันทึกที่ {model_path}")

    # (10) ประเมินบน Test Set (Bulk Predict)
    X_test, y_test = [], []
    for i in range(window_size, len(scaled_df)):
        X_test.append(scaled_df.iloc[i-window_size:i].values)
        y_test.append(scaled_df.iloc[i][value_column])
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # เอาเฉพาะ days_to_remove ช่วงท้าย
    X_test_set = X_test[-days_to_remove:]
    y_test_set = y_test[-days_to_remove:]
    dates_test_set = df_processed.index[-days_to_remove:]

    predictions_scaled = model.predict(X_test_set).ravel()

    # Inverse Transform เฉพาะ value_column
    target_idx = df_processed.columns.get_loc(value_column)
    predictions_full = np.zeros((days_to_remove, df_processed.shape[1]))
    predictions_full[:, target_idx] = predictions_scaled

    y_test_full = np.zeros((days_to_remove, df_processed.shape[1]))
    y_test_full[:, target_idx] = y_test_set

    predictions_inversed = scaler.inverse_transform(predictions_full)[:, target_idx]
    actual_inversed = scaler.inverse_transform(y_test_full)[:, target_idx]

    comparison = pd.DataFrame({
        'Actual': actual_inversed,
        'Predicted': predictions_inversed
    }, index=dates_test_set)

    mae = np.mean(np.abs(actual_inversed - predictions_inversed))
    rmse = np.sqrt(np.mean((actual_inversed - predictions_inversed)**2))
    mape = np.mean(np.abs((actual_inversed - predictions_inversed)/actual_inversed))*100

    print("ผลการประเมิน (Test Set):")
    print(f"MAE = {mae:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"MAPE = {mape:.2f}%")

    comparison_path = 'models/lstm_comparison.csv'
    comparison.to_csv(comparison_path)
    print(f"Comparison บันทึกที่ {comparison_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(comparison['Actual'], label='Actual')
    plt.plot(comparison['Predicted'], label='Predicted')
    plt.title('Actual vs Predicted (Short Trend)')
    plt.xlabel('Date')
    plt.ylabel(value_column)
    plt.legend()
    plt.savefig('models/actual_vs_predicted.png')
    plt.close()

if __name__ == "__main__":
    main()
