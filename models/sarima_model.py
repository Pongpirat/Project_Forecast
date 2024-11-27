import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pmdarima import auto_arima

def sarima_model(data, value_column, days_to_remove, smooth_window=3, seasonal_order=None, order=None):
    # แยกข้อมูลสำหรับฝึกและทดสอบ
    train_data = data.iloc[:-days_to_remove]  # ข้อมูลตั้งแต่ต้นจนถึงก่อนช่วงทดสอบ
    test_data = data.iloc[-days_to_remove:]  # ข้อมูลช่วงทดสอบ

    # Smooth ข้อมูลใน Train Data
    train_data['smoothed'] = train_data[value_column].rolling(window=smooth_window).mean()
    train_data['lag_1'] = train_data['smoothed'].shift(1)
    train_data['lag_7'] = train_data['smoothed'].shift(7)
    train_data = train_data.dropna()

    # Smooth ข้อมูลใน Test Data
    test_data['smoothed'] = test_data[value_column].rolling(window=smooth_window).mean()

    # เติมค่า Missing ใน Test Data
    test_data['smoothed'] = test_data['smoothed'].fillna(method='bfill')  # ใช้ค่าถัดไปเติม

    # เพิ่ม Lag Features ใน Test Data
    test_data['lag_1'] = test_data['smoothed'].shift(1)
    test_data['lag_7'] = test_data['smoothed'].shift(7)

    # เติมค่า Missing ใน Lag Features ของ Test Data
    test_data['lag_1'] = test_data['lag_1'].fillna(method='bfill')
    test_data['lag_7'] = test_data['lag_7'].fillna(method='bfill')

    # ใช้ Auto ARIMA เพื่อหาพารามิเตอร์ที่เหมาะสมหากไม่ได้ระบุ
    if order is None or seasonal_order is None:
        auto_model = auto_arima(
            train_data['smoothed'], 
            seasonal=True, 
            m=7,  # 7 วันสำหรับฤดูกาลรายสัปดาห์
            trace=True,
            error_action='ignore',
            suppress_warnings=True
        )
        order = auto_model.order
        seasonal_order = auto_model.seasonal_order

    # สร้างและฝึกโมเดล SARIMA
    sarima_model = SARIMAX(
        train_data['smoothed'],
        order=order,
        seasonal_order=seasonal_order,
        exog=train_data[['lag_1', 'lag_7']]  # ใช้ Lag Features เป็น Exogenous Variables
    )
    sarima_model_fit = sarima_model.fit(disp=False)

    # พยากรณ์ข้อมูล
    forecast = sarima_model_fit.forecast(
        steps=len(test_data),
        exog=test_data[['lag_1', 'lag_7']]
    )

    # คำนวณค่าความแม่นยำ
    mae = mean_absolute_error(test_data['smoothed'], forecast)
    rmse = np.sqrt(mean_squared_error(test_data['smoothed'], forecast))
    mape = np.mean(np.abs((test_data['smoothed'] - forecast) / test_data['smoothed'])) * 100

    # สร้างตารางเปรียบเทียบ
    comparison = pd.DataFrame({
        'Actual': test_data['smoothed'],
        'Predicted': forecast
    })

    return {
        'comparison': comparison,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'order': order,
        'seasonal_order': seasonal_order
    }