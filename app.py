import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
from datetime import datetime

from data_processing import process_data
from models.exponential_smoothing import exponential_smoothing_model
from models.moving_average import moving_average_model
from models.sarima_model import sarima_model
from models.lstm_inference import inference_lstm_model
from train_lstm import train_lstm_model

def main():
    st.set_page_config(page_title="Compare All Models", layout="wide")
    st.title("Dashboard เปรียบเทียบการพยากรณ์จากทุกโมเดลในหน้าเดียว")
    st.markdown("---")

    # ส่วน Sidebar สำหรับรับค่าพารามิเตอร์ต่าง ๆ
    with st.sidebar:
        st.header("การตั้งค่า")

        # เลือกไฟล์ CSV จากโฟลเดอร์ 'currency'
        preloaded_files_dir = os.path.join(os.getcwd(), 'currency')
        if not os.path.exists(preloaded_files_dir):
            st.error(f"ไม่พบโฟลเดอร์ '{preloaded_files_dir}'")
            st.stop()

        preloaded_files = [f for f in os.listdir(preloaded_files_dir) if f.endswith('.csv')]
        if not preloaded_files:
            st.error(f"ไม่พบไฟล์ CSV ในโฟลเดอร์ '{preloaded_files_dir}'")
            st.stop()

        selected_preloaded_file = st.selectbox("เลือกไฟล์ CSV", preloaded_files)
        preloaded_file_path = os.path.join(preloaded_files_dir, selected_preloaded_file)

        # ระบุจำนวนวันที่ถือเป็น Test Set
        days_to_remove = st.number_input("ระบุจำนวนวันที่ต้องการแยกทดสอบ (Test Set)", min_value=1, value=30)

        # ปิดการใช้งานการพยากรณ์อนาคตข้าง 60 วัน
        # หากต้องการเปิดใช้งานในภายหลัง ให้ทำการ uncomment โค้ดด้านล่างและตั้งค่า forecast_days ตามต้องการ
        # forecast_days = st.number_input("ระบุจำนวนวันที่ต้องการพยากรณ์อนาคต", min_value=1, value=60)
        forecast_days = 0  # ตั้งค่าเป็น 0 เพื่อปิดการพยากรณ์อนาคต

        st.markdown("---")

        # ปุ่มสำหรับฝึกโมเดล LSTM ถ้าไม่พบโมเดล
        base_filename = os.path.splitext(selected_preloaded_file)[0]
        model_folder = os.path.join('models', base_filename)
        model_path = os.path.join(model_folder, 'lstm_model.h5')
        scaler_date_path = os.path.join(model_folder, 'scaler_lstm_date.pkl')
        scaler_target_path = os.path.join(model_folder, 'scaler_lstm_target.pkl')

        if not (os.path.exists(model_path) and os.path.exists(scaler_date_path) and os.path.exists(scaler_target_path)):
            st.warning(f"ไม่พบโมเดล LSTM หรือ Scaler สำหรับไฟล์ '{selected_preloaded_file}'")
            if st.button("ฝึกโมเดล LSTM สำหรับไฟล์นี้"):
                train_success = train_lstm_model(selected_preloaded_file)
                if train_success:
                    st.success(f"ฝึกโมเดล LSTM สำเร็จสำหรับไฟล์ '{selected_preloaded_file}'")
                else:
                    st.error(f"การฝึกโมเดล LSTM ล้มเหลวสำหรับไฟล์ '{selected_preloaded_file}'")

    # โหลดไฟล์ CSV
    try:
        df = pd.read_csv(preloaded_file_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(preloaded_file_path, encoding='cp874')
        except Exception as e:
            st.error(f"ไม่สามารถอ่านไฟล์ได้: {e}")
            st.stop()

    if 'งวด' not in df.columns:
        st.error("ไม่พบคอลัมน์ 'งวด' ในไฟล์ CSV ที่เลือก")
        st.stop()

    # กำหนดค่า value_column ให้ตรงกับข้อมูลดิบ (ไม่ใช่ smoothed_value)
    value_column = 'อัตราขาย'  

    # ประมวลผลข้อมูล (forward fill, reindex, smoothing) 
    # หากไม่ต้องการให้ smooth สำหรับ LSTM สามารถล็อก smoothing_window=3 (ใช้สำหรับโมเดลอื่น) 
    # โดยในส่วนของ LSTM เราจะใช้ column 'อัตราขาย' แทน 'smoothed_value'
    with st.spinner('กำลังประมวลผลข้อมูล...'):
        try:
            processed_df, numeric_cols = process_data(
                df,
                value_column=value_column,
                smoothing_window=3  # ค่า smoothing สำหรับโมเดลอื่นๆ ยังคงใช้ได้
            )
        except ValueError as ve:
            st.error(str(ve))
            st.stop()

    # สร้าง dict เก็บผลลัพธ์จากแต่ละโมเดล
    results = {}

    # ----------------------
    # (1) Exponential Smoothing
    # ----------------------
    with st.spinner("กำลังรันโมเดล Exponential Smoothing..."):
        try:
            exp_result = exponential_smoothing_model(
                data=processed_df,
                value_column=value_column,
                days_to_remove=int(days_to_remove)
            )
            results['Exponential Smoothing'] = exp_result
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดใน Exponential Smoothing: {e}")

    # ----------------------
    # (2) Moving Average (EMA)
    # ----------------------
    with st.spinner("กำลังรันโมเดล Moving Average..."):
        try:
            ma_result = moving_average_model(
                data=processed_df,
                value_column=value_column,
                days_to_remove=int(days_to_remove),
                window_size=30,
                use_ema=True
            )
            results['Moving Average (EMA)'] = ma_result
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดใน Moving Average: {e}")

    # ----------------------
    # (3) SARIMA
    # ----------------------
    with st.spinner("กำลังรันโมเดล SARIMA..."):
        try:
            sarima_result = sarima_model(
                data=processed_df,
                value_column=value_column,
                days_to_remove=int(days_to_remove),
                seasonal_order=(1, 1, 1, 7),
                order=(2, 1, 2)
            )
            results['SARIMA'] = sarima_result
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดใน SARIMA: {e}")

    # ----------------------
    # (4) LSTM
    # ----------------------
    with st.spinner("กำลังรันโมเดล LSTM..."):
        if not (os.path.exists(model_path) and os.path.exists(scaler_date_path) and os.path.exists(scaler_target_path)):
            st.warning(f"ไม่สามารถรันโมเดล LSTM ได้ เนื่องจากไม่พบโมเดลหรือ Scaler สำหรับไฟล์ '{selected_preloaded_file}'")
        else:
            try:
                # ใน LSTM เวลาทำนาย เราใช้ค่า 'อัตราขาย' (raw) แทน smoothed_value
                lstm_result = inference_lstm_model(
                    data=processed_df,
                    value_column=value_column,
                    days_to_remove=int(days_to_remove),
                    model_path=model_path,
                    scaler_date_path=scaler_date_path,
                    scaler_target_path=scaler_target_path,
                    window_size=14,
                    forecast_days=int(forecast_days)  # ใช้ค่า 0 เพื่อปิดการพยากรณ์อนาคต
                )
                results['LSTM'] = lstm_result
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดใน LSTM: {e}")

    if not results:
        st.error("ไม่พบผลลัพธ์จากโมเดลใดเลย")
        st.stop()

    # ------------------------------------------------
    # เปรียบเทียบ Actual vs Predicted ของทุกโมเดล (เฉพาะปี 2024) และตาราง Metrics อยู่แนวเดียวกัน
    # ------------------------------------------------    
    col1, col2 = st.columns([2, 1])  # กำหนดสัดส่วนความกว้าง เช่น 2:1

    with col1:
        fig = go.Figure()

        # Actual สำหรับปี 2024 (ยังใช้ smoothed_value สำหรับโมเดลอื่นๆ)
        actual_series = processed_df['smoothed_value']
        actual_2024 = actual_series[actual_series.index.year == 2024]
        fig.add_trace(go.Scatter(
            x=actual_2024.index,
            y=actual_2024,
            mode='lines',
            name='Actual (Smoothed)',
            line=dict(color='black')
        ))

        # เพิ่ม Prediction ของแต่ละโมเดล (เฉพาะปี 2024)
        for model_name, result_data in results.items():
            comp_df = result_data['comparison']
            comp_2024 = comp_df[comp_df.index.year == 2024]
            fig.add_trace(go.Scatter(
                x=comp_2024.index,
                y=comp_2024['Predicted'],
                mode='lines',
                name=f'Predicted ({model_name})'
            ))

            # ไม่แสดง future_predictions เนื่องจาก forecast_days=0
            # หากเปิดใช้งานในภายหลัง สามารถ uncomment โค้ดด้านล่างได้
            # if 'future_predictions' in result_data:
            #     future_df = result_data['future_predictions']
            #     future_2024 = future_df[future_df.index.year == 2024]
            #     fig.add_trace(go.Scatter(
            #         x=future_2024.index,
            #         y=future_2024['Predicted'],
            #         mode='lines',
            #         name=f'Forecast ({model_name})',
            #         line=dict(dash='dash')
            #     ))

        fig.update_layout(
            title='Actual vs Predicted (All Models) - ปี 2024',
            xaxis_title='Date',
            yaxis_title='อัตราขาย (Smoothed)',
            legend=dict(x=0, y=1),
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Metrics (MAE, RMSE, MAPE)")
        
        metrics_data = []
        for model_name, result_data in results.items():
            row = {
                'Model': model_name,
                'MAE': round(result_data['mae'], 4),
                'RMSE': round(result_data['rmse'], 4),
                'MAPE': f"{round(result_data['mape'], 2)}%"
            }
            metrics_data.append(row)

        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)

    st.subheader("ตารางเปรียบเทียบ Actual vs Predicted")

    combined_df = pd.DataFrame(index=processed_df.index)
    combined_df['Actual (Smoothed)'] = processed_df['smoothed_value']

    for model_name, result_data in results.items():
        comp_df = result_data['comparison']
        predicted_col_name = f'Predicted_{model_name}'
        combined_df = combined_df.join(
            comp_df['Predicted'].rename(predicted_col_name),
            how='left'
        )

    non_nan_columns = ['Actual (Smoothed)'] + [f'Predicted_{model}' for model in results.keys()]
    filtered_combined_df = combined_df.dropna(subset=non_nan_columns)

    st.dataframe(filtered_combined_df, use_container_width=True)

if __name__ == "__main__":
    main()
