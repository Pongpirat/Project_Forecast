import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

def inference_lstm_model(
    data,
    value_column,
    days_to_remove,
    model_path,
    scaler_path,
    window_size,
    forecast_days=0  # จำนวนวันที่ต้องการพยากรณ์อนาคต
):
    """
    Inference แบบ Bulk Predict และ Forecast Future Days
    """
    # ---------- 1) โหลด Scaler ----------
    scaler = joblib.load(scaler_path)

    # ---------- 2) เลือกฟีเจอร์ [value_column, 'diff_1', 'diff_7']
    feature_cols = [value_column, 'diff_1', 'diff_7']
    df_features = data[feature_cols].copy()

    # ---------- 3) สเกลข้อมูล ----------
    scaled_data = scaler.transform(df_features)
    scaled_df = pd.DataFrame(scaled_data, index=df_features.index, columns=df_features.columns)

    # ---------- 4) โหลดโมเดล LSTM ----------
    model = load_model(model_path)

    # ---------- 5) สร้าง sequence ทั้งหมด ----------
    X_test, y_test = [], []
    for i in range(window_size, len(scaled_df)):
        X_test.append(scaled_df.iloc[i-window_size:i].values)
        y_test.append(scaled_df.iloc[i][value_column])
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # ---------- 6) เอาเฉพาะ days_to_remove ช่วงท้าย ----------
    X_test_set = X_test[-days_to_remove:]
    y_test_set = y_test[-days_to_remove:]
    test_dates = data.index[-days_to_remove:]

    # ---------- 7) ทำนาย ----------
    predictions_scaled = model.predict(X_test_set).ravel()

    # ---------- 8) Inverse Transform ----------
    target_idx = feature_cols.index(value_column)
    predictions_full = np.zeros((days_to_remove, len(feature_cols)))
    predictions_full[:, target_idx] = predictions_scaled

    y_test_full = np.zeros((days_to_remove, len(feature_cols)))
    y_test_full[:, target_idx] = y_test_set

    predictions_inversed = scaler.inverse_transform(predictions_full)[:, target_idx]
    actual_inversed = scaler.inverse_transform(y_test_full)[:, target_idx]

    # ---------- 9) สร้าง DataFrame เปรียบเทียบ ----------
    comparison = pd.DataFrame({
        'Actual': actual_inversed,
        'Predicted': predictions_inversed
    }, index=test_dates)

    # ---------- 10) คำนวณ Metrics ----------
    mae = np.mean(np.abs(comparison['Actual'] - comparison['Predicted']))
    rmse = np.sqrt(np.mean((comparison['Actual'] - comparison['Predicted'])**2))
    mape = np.mean(np.abs((comparison['Actual'] - comparison['Predicted']) / comparison['Actual'])) * 100

    # ---------- 11) Forecast Future Days ----------
    if forecast_days > 0:
        # Initialize list to collect future predictions
        future_predictions_scaled = []
        future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='D')

        # Start with the last window_size rows from scaled_df
        last_sequence = scaled_df.iloc[-window_size:].copy()

        for i in range(forecast_days):
            # Prepare input sequence
            input_seq = last_sequence.values.reshape(1, window_size, len(feature_cols))

            # Predict
            pred_scaled = model.predict(input_seq)[0][0]
            future_predictions_scaled.append(pred_scaled)

            # Update diff features based on the prediction
            # diff_1 = pred(t) - pred(t-1)
            new_diff_1 = pred_scaled - last_sequence.iloc[-1][value_column]

            # diff_7 = pred(t) - pred(t-7)
            if window_size >= 7:
                new_diff_7 = pred_scaled - last_sequence.iloc[-7][value_column]
            else:
                new_diff_7 = 0  # fallback

            # Create new row
            new_row = [pred_scaled, new_diff_1, new_diff_7]

            # Append new row to the sequence and drop the first row to maintain window_size
            new_row_df = pd.DataFrame([new_row], columns=feature_cols, index=[future_dates[i]])
            last_sequence = pd.concat([last_sequence, new_row_df]).iloc[-window_size:]

        # Convert predictions_scaled to inverse transformed values
        future_predictions_full = np.zeros((forecast_days, len(feature_cols)))
        future_predictions_full[:, target_idx] = future_predictions_scaled
        future_predictions_inversed = scaler.inverse_transform(future_predictions_full)[:, target_idx]

        # Create DataFrame for future predictions
        future_comparison = pd.DataFrame({
            'Predicted': future_predictions_inversed
        }, index=future_dates)

        # Add future predictions to result
        comparison = comparison.append(future_comparison)

    # ---------- 12) สร้างผลลัพธ์ ----------
    result = {
        'comparison': comparison,
        'mae': mae,
        'rmse': rmse,
        'mape': mape
    }

    if forecast_days > 0:
        future_comparison = comparison.iloc[-forecast_days:]
        result['future_predictions'] = future_comparison

    return result
