# train_lstm_full.py
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from data_processing import process_data

def train_lstm_full_model(selected_preloaded_file):
    """
    ฝึกโมเดล LSTM โดยใช้ข้อมูลทั้งหมด (ไม่ตัด 30 วันออก) จากไฟล์ CSV ที่เลือก
    และบันทึกโมเดล, Scaler พร้อมทั้งประเมินผล (บน Validation Set)
    """
    try:
        preloaded_files_dir = os.path.join(os.getcwd(), 'currency')
        csv_file_path = os.path.join(preloaded_files_dir, selected_preloaded_file)
        df = pd.read_csv(csv_file_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_file_path, encoding='cp874')
        except Exception as e:
            print(f"ไม่สามารถอ่านไฟล์ได้: {e}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

    if 'งวด' not in df.columns:
        print("ไม่พบคอลัมน์ 'งวด' ในไฟล์ CSV ที่เลือก")
        return False

    # 1) ประมวลผลข้อมูล (reindex, forward fill)
    value_column = 'อัตราขาย'
    df_processed, numeric_cols = process_data(df, value_column)

    # 2) สร้าง Lag features จากข้อมูลดิบ (อัตราขาย)
    df_processed['lag_1'] = df_processed[value_column].shift(1)
    df_processed['lag_2'] = df_processed[value_column].shift(2)
    df_processed.dropna(inplace=True)

    # ใช้ข้อมูลทั้งหมดที่ผ่านการประมวลผล (ไม่แบ่งแยก Train/Test)
    # เราจะใช้ข้อมูลทั้งหมดสำหรับการ fit Scaler เพื่อให้ครอบคลุมความผันผวนในช่วงท้าย
    # กำหนด features
    date_features = ['day', 'month', 'year', 'week']
    target_features = [value_column, 'lag_1', 'lag_2']
    all_features = date_features + target_features

    # 3) Fit Scaler บนข้อมูลทั้งหมดที่ผ่านการประมวลผล
    scaler_date = MinMaxScaler()
    scaler_target = MinMaxScaler()

    full_date_scaled = scaler_date.fit_transform(df_processed[date_features])
    full_target_scaled = scaler_target.fit_transform(df_processed[target_features])
    full_scaled = np.hstack([full_date_scaled, full_target_scaled])
    full_scaled_df = pd.DataFrame(full_scaled, index=df_processed.index, columns=all_features)

    # 4) สร้าง Sequence สำหรับ Train จากข้อมูลทั้งหมด
    window_size = 14
    X, y, seq_dates = [], [], []
    train_values = full_scaled_df.values
    # ตำแหน่งของ value_column ใน target_features คือ index หลังจาก date_features
    idx_target = len(date_features)

    for i in range(window_size, len(train_values)):
        X.append(train_values[i - window_size:i, :])
        y.append(train_values[i, idx_target])
        seq_dates.append(df_processed.index[i])
    X = np.array(X)
    y = np.array(y)
    seq_dates = np.array(seq_dates)

    # แบ่ง Train/Validation แบบ 80:20
    split_idx = int(len(X) * 0.8)
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_val = X[split_idx:]
    y_val = y[split_idx:]
    seq_dates_train = seq_dates[:split_idx]
    seq_dates_val = seq_dates[split_idx:]

    # 5) สร้างโมเดล LSTM (ใช้สถาปัตยกรรมเดียวกับใน train_lstm.py)
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

    print("กำลังฝึกโมเดล LSTM (Full Data)...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    print("ฝึกโมเดลเสร็จสิ้น")

    # 6) บันทึกโมเดลและ Scaler
    base_filename = os.path.splitext(selected_preloaded_file)[0]
    model_folder = os.path.join('models', base_filename, 'full_data')
    os.makedirs(model_folder, exist_ok=True)

    model_path = os.path.join(model_folder, 'lstm_model_full.h5')
    model.save(model_path)
    print(f"โมเดลบันทึกที่ {model_path}")

    scaler_date_path = os.path.join(model_folder, 'scaler_lstm_date_full.pkl')
    scaler_target_path = os.path.join(model_folder, 'scaler_lstm_target_full.pkl')
    joblib.dump(scaler_date, scaler_date_path)
    joblib.dump(scaler_target, scaler_target_path)
    print(f"Scaler สำหรับ Date Features บันทึกที่ {scaler_date_path}")
    print(f"Scaler สำหรับ Target Features บันทึกที่ {scaler_target_path}")

    # 7) Plot Loss (Train และ Validation)
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss (Full Data)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    loss_plot_path = os.path.join(model_folder, 'loss_plot_full.png')
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Plot Loss บันทึกที่ {loss_plot_path}")

    # 8) ประเมินโมเดลบน Validation Set
    preds_val_scaled = model.predict(X_val).ravel()
    predictions_full = np.zeros((len(preds_val_scaled), len(all_features)))
    predictions_full[:, idx_target] = preds_val_scaled
    pred_inv = scaler_target.inverse_transform(predictions_full[:, len(date_features):])[:, 0]

    y_val_full = np.zeros((len(y_val), len(all_features)))
    y_val_full[:, idx_target] = y_val
    actual_inv = scaler_target.inverse_transform(y_val_full[:, len(date_features):])[:, 0]

    mae = np.mean(np.abs(actual_inv - pred_inv))
    rmse = np.sqrt(np.mean((actual_inv - pred_inv) ** 2))
    mape = np.mean(np.abs((actual_inv - pred_inv) / actual_inv)) * 100

    print("ผลการประเมิน (Validation Set):")
    print(f"MAE = {mae:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"MAPE = {mape:.2f}%")

    # สร้าง DataFrame สำหรับเปรียบเทียบ Actual vs Predicted
    comparison = pd.DataFrame({
        'Actual': actual_inv,
        'Predicted': pred_inv
    }, index=seq_dates_val)

    comparison_path = os.path.join(model_folder, 'lstm_comparison_full.csv')
    comparison.to_csv(comparison_path)
    print(f"Comparison บันทึกที่ {comparison_path}")

    # วาดกราฟ Actual vs Predicted
    plt.figure(figsize=(10, 6))
    plt.plot(comparison.index, comparison['Actual'], label='Actual')
    plt.plot(comparison.index, comparison['Predicted'], label='Predicted')
    plt.title('Actual vs Predicted (Full Data) - Validation')
    plt.xlabel('Date')
    plt.ylabel(value_column)
    plt.legend()
    actual_vs_predicted_plot_path = os.path.join(model_folder, 'actual_vs_predicted_full.png')
    plt.savefig(actual_vs_predicted_plot_path)
    plt.close()
    print(f"Plot Actual vs Predicted บันทึกที่ {actual_vs_predicted_plot_path}")

    return True

if __name__ == "__main__":
    # ตัวอย่างการฝึกโมเดล LSTM สำหรับไฟล์ 'AUD.csv' โดยใช้ข้อมูลทั้งหมด (ไม่ตัด 30 วันออก)
    success = train_lstm_full_model('HKD.csv')
    if success:
        print("การฝึกโมเดลด้วยข้อมูลทั้งหมด (Full Data) สำเร็จ")
    else:
        print("การฝึกโมเดลด้วยข้อมูลทั้งหมด (Full Data) ล้มเหลว")
