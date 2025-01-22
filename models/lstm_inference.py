import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

def inference_lstm_model(
    data,
    value_column,
    days_to_remove,
    model_path,
    scaler_date_path,
    scaler_target_path,
    window_size,
    forecast_days=0
):
    """
    Inference สำหรับโมเดล LSTM (รวม Recursive Forecast)
    โดยใช้ 2 Scaler (date + target) และใช้ข้อมูลดิบ (อัตราขาย) แทนการใช้ smoothing
    """
    # สร้าง lag ถ้ายังไม่มี (ปกติใน data ควรมีแล้ว แต่กันพลาด)
    if 'lag_1' not in data.columns:
        data['lag_1'] = data[value_column].shift(1)
    if 'lag_2' not in data.columns:
        data['lag_2'] = data[value_column].shift(2)
    data.dropna(inplace=True)

    date_features = ['day', 'month', 'year', 'week']
    target_features = [value_column, 'lag_1', 'lag_2']

    scaler_date = joblib.load(scaler_date_path)
    scaler_target = joblib.load(scaler_target_path)
    model = load_model(model_path)

    # ------------------------------------------------
    # 1) สร้าง Test Set
    # ------------------------------------------------
    df_test = data.iloc[-days_to_remove:]
    df_all = data

    test_date_scaled = scaler_date.transform(df_test[date_features])
    test_target_scaled = scaler_target.transform(df_test[target_features])
    test_scaled = np.hstack([test_date_scaled, test_target_scaled])

    all_features = date_features + target_features
    test_scaled_df = pd.DataFrame(test_scaled, index=df_test.index, columns=all_features)

    df_before_test = data.iloc[: -days_to_remove]
    if len(df_before_test) < window_size:
        raise ValueError("ข้อมูลไม่พอสร้าง sequence")

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

    result = {
        'comparison': comparison,
        'mae': mae,
        'rmse': rmse,
        'mape': mape
    }

    # ------------------------------------------------
    # 2) Recursive Forecasting (forecast_days)
    # ------------------------------------------------
    if forecast_days > 0:
        df_forecast = data.copy()
        full_date_scaled = scaler_date.transform(df_forecast[date_features])
        full_target_scaled = scaler_target.transform(df_forecast[target_features])
        full_scaled = np.hstack([full_date_scaled, full_target_scaled])
        full_scaled_df = pd.DataFrame(full_scaled, index=df_forecast.index, columns=all_features)

        last_sequence = full_scaled_df.iloc[-window_size:].copy()

        future_dates = pd.date_range(start=df_forecast.index[-1] + pd.Timedelta(days=1),
                                     periods=forecast_days, freq='D')

        future_features_date = pd.DataFrame(index=future_dates)
        future_features_date['day'] = future_features_date.index.day
        future_features_date['month'] = future_features_date.index.month
        future_features_date['year'] = future_features_date.index.year
        future_features_date['week'] = future_features_date.index.isocalendar().week

        future_pred_list = []

        for i in range(forecast_days):
            row_date_unscaled = future_features_date.iloc[i]  # day,month,year,week
            row_date_scaled = scaler_date.transform([row_date_unscaled.values])  # shape=(1,4)

            input_X = last_sequence.values.reshape(1, window_size, len(all_features))
            pred_scaled_value = model.predict(input_X)[0][0]

            # อัปเดต lag:
            prev_lag_1_s = last_sequence.iloc[-1, idx_target + 1]  # lag_1 ของแถวสุดท้าย
            new_lag_1_s = pred_scaled_value
            new_lag_2_s = prev_lag_1_s

            row_target_scaled = np.array([[pred_scaled_value, new_lag_1_s, new_lag_2_s]], dtype=float)
            new_scaled_row = np.hstack([row_date_scaled, row_target_scaled])  # shape=(1,7)

            new_sequence = np.vstack([last_sequence.values[1:], new_scaled_row])
            last_sequence = pd.DataFrame(new_sequence, columns=all_features)

            tmp_full = np.zeros((1, len(all_features)))
            tmp_full[0, idx_target] = pred_scaled_value
            tmp_target_part = tmp_full[:, len(date_features):]  # [อัตราขาย, lag_1, lag_2]
            pred_inv_value = scaler_target.inverse_transform(tmp_target_part)[0, 0]

            future_pred_list.append(pred_inv_value)

        future_comparison = pd.DataFrame({
            'Predicted': future_pred_list
        }, index=future_dates)

        result['future_predictions'] = future_comparison

    return result
