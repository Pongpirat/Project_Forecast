import numpy as np
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from data_processing import process_data

def train_lstm_model(selected_preloaded_file):
    """
    ฝึกโมเดล LSTM สำหรับไฟล์ CSV ที่เลือกและบันทึกโมเดลและ Scaler ลงในโฟลเดอร์เฉพาะ
    """
    try:
        # --------------------------------------------------------
        # 1) เลือกไฟล์ CSV
        # --------------------------------------------------------
        preloaded_files_dir = os.path.join(os.getcwd(), 'currency')
        csv_file_path = os.path.join(preloaded_files_dir, selected_preloaded_file)

        df = pd.read_csv(csv_file_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_file_path, encoding='cp874')
        except Exception as e:
            print(f"ไม่สามารถอ่านไฟล์ได้: {e}")
            return False

    if 'งวด' not in df.columns:
        print("ไม่พบคอลัมน์ 'งวด' ในไฟล์ CSV ที่เลือก")
        return False

    # --------------------------------------------------------
    # 2) ประมวลผลข้อมูล (reindex, forward fill, smoothing)
    # --------------------------------------------------------
    value_column = 'อัตราขาย'
    df_processed, numeric_cols = process_data(df, value_column, smoothing_window=3)

    # --------------------------------------------------------
    # 3) สร้าง Lag features จากข้อมูลดิบ (อัตราขาย)
    # --------------------------------------------------------
    df_processed['lag_1'] = df_processed[value_column].shift(1)
    df_processed['lag_2'] = df_processed[value_column].shift(2)
    df_processed.dropna(inplace=True)

    # --------------------------------------------------------
    # 4) แยกฟีเจอร์เป็น 2 กลุ่ม: date_features, target_features
    # --------------------------------------------------------
    date_features = ['day', 'month', 'year', 'week']
    target_features = [value_column, 'lag_1', 'lag_2']

    days_to_remove = 30
    df_train = df_processed.iloc[:-days_to_remove]
    df_test = df_processed.iloc[-days_to_remove:]

    # --------------------------------------------------------
    # 5) สร้าง Scaler 2 ตัว และ fit แยก
    # --------------------------------------------------------
    scaler_date = MinMaxScaler()
    scaler_target = MinMaxScaler()

    # fit เฉพาะช่วง train
    train_date_scaled = scaler_date.fit_transform(df_train[date_features])
    train_target_scaled = scaler_target.fit_transform(df_train[target_features])

    # รวม date + target เป็น train_scaled
    train_scaled = np.hstack([train_date_scaled, train_target_scaled])

    all_features = date_features + target_features
    train_scaled_df = pd.DataFrame(train_scaled, index=df_train.index, columns=all_features)

    # สร้างโฟลเดอร์สำหรับโมเดล
    base_filename = os.path.splitext(selected_preloaded_file)[0]
    model_folder = os.path.join('models', base_filename)
    os.makedirs(model_folder, exist_ok=True)

    scaler_date_path = os.path.join(model_folder, 'scaler_lstm_date.pkl')
    scaler_target_path = os.path.join(model_folder, 'scaler_lstm_target.pkl')

    # เซฟ scaler
    joblib.dump(scaler_date, scaler_date_path)
    joblib.dump(scaler_target, scaler_target_path)
    print(f"Scaler สำหรับ Date Features บันทึกที่ {scaler_date_path}")
    print(f"Scaler สำหรับ Target Features บันทึกที่ {scaler_target_path}")

    # --------------------------------------------------------
    # 6) สร้าง Sequence สำหรับ Train
    #    window_size = 14
    # --------------------------------------------------------
    window_size = 14
    X_train, y_train = [], []
    train_values = train_scaled_df.values  # shape = (len(df_train), 7)

    # ใน all_features ส่วน target เริ่มที่ index len(date_features)
    idx_target = len(date_features)

    for i in range(window_size, len(train_values)):
        X_train.append(train_values[i - window_size:i, :])
        # ให้ y_train เป็นค่าจากคอลัมน์ อัตราขาย (raw value scaled)
        y_train.append(train_values[i, idx_target])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # --------------------------------------------------------
    # 7) แบ่ง Train/Validation
    # --------------------------------------------------------
    train_size = int(len(X_train) * 0.8)
    X_val = X_train[train_size:]
    y_val = y_train[train_size:]
    X_train2 = X_train[:train_size]
    y_train2 = y_train[:train_size]

    # --------------------------------------------------------
    # 8) สร้างโมเดล LSTM
    # --------------------------------------------------------
    model = Sequential([
        Bidirectional(LSTM(256, return_sequences=True, activation='relu'), 
                      input_shape=(window_size, train_values.shape[1])),
        Dropout(0.3),
        LSTM(128, return_sequences=True, activation='relu'),
        Dropout(0.3),
        LSTM(64, return_sequences=False, activation='relu'),
        Dropout(0.3),
        Dense(1)
    ])

    optimizer = Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

    epochs = 500
    batch_size = 32

    print("กำลังฝึกโมเดล LSTM...")
    history = model.fit(
        X_train2, y_train2,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    print("ฝึกโมเดลเสร็จสิ้น")

    # --------------------------------------------------------
    # 9) บันทึกโมเดล + Plot Loss
    # --------------------------------------------------------
    model_path = os.path.join(model_folder, 'lstm_model.h5')
    model.save(model_path)
    print(f"โมเดลบันทึกที่ {model_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    loss_plot_path = os.path.join(model_folder, 'loss_plot.png')
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Plot Loss บันทึกที่ {loss_plot_path}")

    # --------------------------------------------------------
    # 10) ประเมินบน Test Set
    # --------------------------------------------------------
    test_date_scaled = scaler_date.transform(df_test[date_features])
    test_target_scaled = scaler_target.transform(df_test[target_features])
    test_scaled = np.hstack([test_date_scaled, test_target_scaled])

    test_scaled_df = pd.DataFrame(test_scaled, index=df_test.index, columns=all_features)

    df_before_test = df_processed.iloc[: -days_to_remove]
    if len(df_before_test) < window_size:
        print("ข้อมูลไม่พอสร้าง sequence")
        return False

    # สร้าง scaled สำหรับช่วงก่อน test (เอา window_size แถวสุดท้าย)
    last_train = df_before_test.iloc[-window_size:]
    last_train_date_scaled = scaler_date.transform(last_train[date_features])
    last_train_target_scaled = scaler_target.transform(last_train[target_features])
    last_train_scaled = np.hstack([last_train_date_scaled, last_train_target_scaled])

    combined_scaled = np.vstack([last_train_scaled, test_scaled])
    combined_index = last_train.index.tolist() + test_scaled_df.index.tolist()
    combined_df = pd.DataFrame(combined_scaled, index=combined_index, columns=all_features)

    X, y = [], []
    idx_target = len(date_features)  # ตำแหน่งของ value_column ใน target_features

    for i in range(window_size, len(combined_df)):
        X.append(combined_df.values[i - window_size:i, :])
        y.append(combined_df.values[i, idx_target])

    X = np.array(X)
    y = np.array(y)

    test_index = combined_df.index[window_size:][-days_to_remove:]
    X_test = X[-days_to_remove:]
    y_test = y[-days_to_remove:]

    preds_test_scaled = model.predict(X_test).ravel()

    # สร้าง array เต็มเพื่อ inverse
    predictions_full = np.zeros((len(preds_test_scaled), len(all_features)))
    predictions_full[:, idx_target] = preds_test_scaled

    y_test_full = np.zeros((len(y_test), len(all_features)))
    y_test_full[:, idx_target] = y_test

    pred_target_part = predictions_full[:, len(date_features):]  # [อัตราขาย, lag_1, lag_2]
    actual_target_part = y_test_full[:, len(date_features):]

    pred_inv = scaler_target.inverse_transform(pred_target_part)[:, 0]
    actual_inv = scaler_target.inverse_transform(actual_target_part)[:, 0]

    comparison = pd.DataFrame({
        'Actual': actual_inv,
        'Predicted': pred_inv
    }, index=test_index)

    mae = np.mean(np.abs(comparison['Actual'] - comparison['Predicted']))
    rmse = np.sqrt(np.mean((comparison['Actual'] - comparison['Predicted'])**2))
    mape = np.mean(np.abs((comparison['Actual'] - comparison['Predicted']) / comparison['Actual'])) * 100

    print("ผลการประเมิน (Test Set):")
    print(f"MAE = {mae:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"MAPE = {mape:.2f}%")

    comparison_path = os.path.join(model_folder, 'lstm_comparison.csv')
    comparison.to_csv(comparison_path)
    print(f"Comparison บันทึกที่ {comparison_path}")

    # วาดกราฟ
    plt.figure(figsize=(10, 6))
    plt.plot(comparison['Actual'], label='Actual')
    plt.plot(comparison['Predicted'], label='Predicted')
    plt.title('Actual vs Predicted (Raw Trend) - Test Set')
    plt.xlabel('Date')
    plt.ylabel(value_column)
    plt.legend()
    actual_vs_predicted_plot_path = os.path.join(model_folder, 'actual_vs_predicted.png')
    plt.savefig(actual_vs_predicted_plot_path)
    plt.close()
    print(f"Plot Actual vs Predicted บันทึกที่ {actual_vs_predicted_plot_path}")

    return True

if __name__ == "__main__":
    # ตัวอย่างการฝึกโมเดลสำหรับไฟล์ 'AUD.csv'
    train_success = train_lstm_model('VND.csv')
    if train_success:
        print("การฝึกโมเดลสำเร็จ")
    else:
        print("การฝึกโมเดลล้มเหลว")
