# app.py
import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go

from datetime import datetime

# import ฟังก์ชันจากโมดูลต่าง ๆ
from data_processing import process_data
from models.exponential_smoothing import exponential_smoothing_model
from models.moving_average import moving_average_model
from models.sarima_model import sarima_model
from models.lstm_inference import inference_lstm_model
from train_lstm import train_lstm_model

# ----------------------------------------------------------
# 1) ใช้ Streamlit Caching ป้องกันการโหลดข้อมูลซ้ำ
# ----------------------------------------------------------
@st.cache_data
def load_data(preloaded_file_path: str) -> pd.DataFrame:
    """
    โหลดไฟล์ CSV โดยพยายามอ่านด้วย encoding='utf-8' ก่อน
    ถ้าไม่ได้จึงลอง cp874
    """
    try:
        df = pd.read_csv(preloaded_file_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(preloaded_file_path, encoding='cp874')
    return df

# ----------------------------------------------------------
# 2) ใช้ Streamlit Caching รวบการประมวลผล + รันโมเดลทั้งหมด
# ----------------------------------------------------------
@st.cache_data
def run_all_models(
    df: pd.DataFrame,
    value_column: str,
    days_to_remove: int,
    forecast_days: int,
    model_path: str,
    scaler_date_path: str,
    scaler_target_path: str
):
    """
    ประมวลผลข้อมูล + รันโมเดลทุกตัว (Exponential Smoothing, MA, SARIMA, LSTM)
    แล้วคืน dict ที่เก็บผลลัพธ์ของแต่ละโมเดล
    """
    results = {}

    # -------------------------------
    # ประมวลผลข้อมูล (process_data)
    # -------------------------------
    processed_df, numeric_cols = process_data(
        df,
        value_column=value_column,
        smoothing_window=3  # smoothing สำหรับโมเดลอื่น ๆ
    )

    # --------------------------------------
    # (1) Exponential Smoothing
    # --------------------------------------
    try:
        exp_result = exponential_smoothing_model(
            data=processed_df,
            value_column=value_column,
            days_to_remove=int(days_to_remove)
        )
        results['Exponential Smoothing'] = exp_result
    except Exception as e:
        results['Exponential Smoothing'] = None
        print(f"[ERROR] Exponential Smoothing: {e}")

    # --------------------------------------
    # (2) Moving Average (EMA)
    # --------------------------------------
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
        results['Moving Average (EMA)'] = None
        print(f"[ERROR] Moving Average: {e}")

    # --------------------------------------
    # (3) SARIMA
    # --------------------------------------
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
        results['SARIMA'] = None
        print(f"[ERROR] SARIMA: {e}")

    # --------------------------------------
    # (4) LSTM
    # --------------------------------------
    if os.path.exists(model_path) and os.path.exists(scaler_date_path) and os.path.exists(scaler_target_path):
        try:
            # ใน LSTM เวลาทำนาย ใช้ค่า 'อัตราขาย' (raw) แทน smoothed_value
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
            results['LSTM'] = None
            print(f"[ERROR] LSTM: {e}")
    else:
        results['LSTM'] = None

    return processed_df, results


def main():
    st.set_page_config(page_title="Compare All Models", layout="wide")
    st.title("Dashboard เปรียบเทียบการพยากรณ์จากทุกโมเดลในหน้าเดียว")
    st.markdown("---")

    # -------------------------------------------------------
    # ส่วน Sidebar สำหรับรับค่าพารามิเตอร์ต่าง ๆ
    # -------------------------------------------------------
    with st.sidebar:
        st.header("การตั้งค่า")

        # 1) เลือกไฟล์ CSV จากโฟลเดอร์ 'currency'
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

        # 2) ระบุจำนวนวันที่ถือเป็น Test Set
        days_to_remove = st.number_input("ระบุจำนวนวันที่ต้องการแยกทดสอบ (Test Set)", min_value=1, value=30)

        # 3) กำหนดจำนวนวันที่ต้องการพยากรณ์อนาคต (ถ้าต้องการเปิดใช้)
        # forecast_days = st.number_input("ระบุจำนวนวันที่ต้องการพยากรณ์อนาคต", min_value=1, value=60)
        # หากต้องการปิดการพยากรณ์ ให้เซตเป็น 0
        forecast_days = 0

        st.markdown("---")

        # 4) ตรวจสอบโมเดล LSTM (ถ้าไม่พบให้ฝึก)
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

    # -------------------------------------------------------
    # 5) โหลดไฟล์ CSV (ใช้ cache) + ตรวจสอบคอลัมน์
    # -------------------------------------------------------
    df = load_data(preloaded_file_path)
    if 'งวด' not in df.columns:
        st.error("ไม่พบคอลัมน์ 'งวด' ในไฟล์ CSV ที่เลือก")
        st.stop()

    # -------------------------------------------------------
    # 6) รันโมเดลทั้งหมด (ใช้ cache) => ได้ (processed_df, results)
    # -------------------------------------------------------
    value_column = 'อัตราขาย'  # ตัวแปรเป้าหมาย
    with st.spinner('กำลังประมวลผลข้อมูลและรันโมเดล...'):
        processed_df, results = run_all_models(
            df=df,
            value_column=value_column,
            days_to_remove=int(days_to_remove),
            forecast_days=int(forecast_days),
            model_path=model_path,
            scaler_date_path=scaler_date_path,
            scaler_target_path=scaler_target_path
        )

    # -------------------------------------------------------
    # 7) ส่วนของการเลือกช่วงปี (Slider) => *ไม่* ไปรันโมเดลซ้ำ
    # -------------------------------------------------------
    years = sorted(processed_df.index.year.unique())
    if not years:
        st.error("ไม่มีข้อมูลปีในชุดข้อมูลที่ประมวลผล")
        st.stop()

    # กำหนด default year = 2024 ถ้ามีในข้อมูล
    default_year = 2024 if 2024 in years else years[-1]

    with st.sidebar:
        selected_year_range = st.slider(
            "เลือกช่วงปีที่ต้องการแสดง",
            min_value=min(years),
            max_value=max(years),
            value=(default_year, default_year),
            step=1
        )
        selected_years = list(range(selected_year_range[0], selected_year_range[1] + 1))

    # -------------------------------------------------------
    # 8) สร้างกราฟเปรียบเทียบ Actual vs Predicted
    # -------------------------------------------------------
    col1, col2 = st.columns([2, 1])

    with col1:
        fig = go.Figure()

        # Actual (Smoothed) ตามปีที่เลือก
        actual_series = processed_df['smoothed_value']
        actual_selected = actual_series[actual_series.index.year.isin(selected_years)]
        fig.add_trace(go.Scatter(
            x=actual_selected.index,
            y=actual_selected,
            mode='lines',
            name='Actual (Smoothed)',
            line=dict(color='black')
        ))

        # เพิ่ม Prediction ของแต่ละโมเดล
        for model_name, result_data in results.items():
            if not result_data:
                continue  # ถ้าโมเดลรันไม่สำเร็จให้ข้าม
            comp_df = result_data['comparison']
            comp_selected = comp_df[comp_df.index.year.isin(selected_years)]
            fig.add_trace(go.Scatter(
                x=comp_selected.index,
                y=comp_selected['Predicted'],
                mode='lines',
                name=f'Predicted ({model_name})'
            ))

            # ถ้ามี future_predictions (forecast_days > 0) สามารถเพิ่มเส้น dashed ได้ตามต้องการ
            # if 'future_predictions' in result_data:
            #     future_df = result_data['future_predictions']
            #     future_selected = future_df[future_df.index.year.isin(selected_years)]
            #     fig.add_trace(go.Scatter(
            #         x=future_selected.index,
            #         y=future_selected['Predicted'],
            #         mode='lines',
            #         name=f'Forecast ({model_name})',
            #         line=dict(dash='dash')
            #     ))

        fig.update_layout(
            title=f'Actual vs Predicted (All Models) - ปี {selected_year_range[0]} ถึง {selected_year_range[1]}',
            xaxis_title='Date',
            yaxis_title='อัตราขาย (Smoothed)',
            legend=dict(x=0, y=1),
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------
    # 9) แสดงตาราง Metrics (MAE, RMSE, MAPE)
    # -------------------------------------------------------
    with col2:
        st.markdown("### Metrics (MAE, RMSE, MAPE)")

        metrics_data = []
        for model_name, result_data in results.items():
            if not result_data:
                # โมเดลมี error หรือไม่สำเร็จ
                metrics_data.append({
                    'Model': model_name,
                    'MAE': '-',
                    'RMSE': '-',
                    'MAPE': '-'
                })
                continue

            metrics_data.append({
                'Model': model_name,
                'MAE': round(result_data['mae'], 4),
                'RMSE': round(result_data['rmse'], 4),
                'MAPE': f"{round(result_data['mape'], 2)}%"
            })

        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)

    # -------------------------------------------------------
    # 10) สรุปตารางเปรียบเทียบ Actual vs Predicted
    # -------------------------------------------------------
    st.subheader("ตารางเปรียบเทียบ Actual vs Predicted")

    combined_df = pd.DataFrame(index=processed_df.index)
    combined_df['Actual (Smoothed)'] = processed_df['smoothed_value']

    for model_name, result_data in results.items():
        if not result_data:
            continue
        comp_df = result_data['comparison']
        predicted_col_name = f'Predicted_{model_name}'
        combined_df = combined_df.join(
            comp_df['Predicted'].rename(predicted_col_name),
            how='left'
        )

    non_nan_columns = ['Actual (Smoothed)'] + [f'Predicted_{model}' for model in results.keys() if results[model] is not None]
    filtered_combined_df = combined_df.dropna(subset=non_nan_columns)
    filtered_combined_df_selected = filtered_combined_df[filtered_combined_df.index.year.isin(selected_years)]

    st.dataframe(filtered_combined_df_selected, use_container_width=True)


if __name__ == "__main__":
    main()
