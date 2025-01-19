import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go

from data_processing import process_data
from models.exponential_smoothing import exponential_smoothing_model
from models.moving_average import moving_average_model
from models.sarima_model import sarima_model
from models.lstm_inference import inference_lstm_model

def main():
    st.set_page_config(page_title="Compare All Models", layout="wide")
    st.title("Dashboard เปรียบเทียบการพยากรณ์จากทุกโมเดลในหน้าเดียว")
    st.markdown("---")

    # ---------- Sidebar: เลือกไฟล์ CSV & กำหนด Days to Remove ----------
    with st.sidebar:
        st.header("การตั้งค่า")

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
        days_to_remove = st.number_input("ระบุจำนวนวันที่ต้องการแยกทดสอบ", min_value=1, value=30)

    # ---------- โหลดและตรวจสอบไฟล์ CSV ----------
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

    # ---------- ประมวลผลข้อมูล (Process Data) ----------
    with st.spinner('กำลังประมวลผลข้อมูล...'):
        processed_df, numeric_cols = process_data(df)

    # ---------- เลือกคอลัมน์ที่ต้องการพยากรณ์ ----------
    value_column = st.sidebar.selectbox("เลือกคอลัมน์ค่าที่ต้องการทำนาย", options=numeric_cols)

    if not value_column:
        st.warning("กรุณาเลือกคอลัมน์ค่าที่ต้องการทำนาย")
        st.stop()

    # ---------- สร้าง Dictionary เก็บผลลัพธ์ของแต่ละโมเดล ----------
    results = {}
    
    # ---------- 1) Exponential Smoothing ----------
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

    # ---------- 2) Moving Average (EMA) ----------
    with st.spinner("กำลังรันโมเดล Moving Average..."):
        try:
            ma_result = moving_average_model(
                data=processed_df,
                value_column=value_column,
                days_to_remove=int(days_to_remove),
                window_size=30,  # สามารถปรับได้
                use_ema=True
            )
            results['Moving Average (EMA)'] = ma_result
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดใน Moving Average: {e}")

    # ---------- 3) SARIMA ----------
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

    # ---------- 4) LSTM ----------
    with st.spinner("กำลังรันโมเดล LSTM..."):
        # กำหนด path ของไฟล์โมเดลและ scaler
        model_path = os.path.join('models', 'lstm_model.h5')
        scaler_path = os.path.join('models', 'scaler_lstm.pkl')

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            st.warning("ไม่พบไฟล์โมเดล LSTM หรือ Scaler. กรุณาฝึกโมเดล LSTM ให้เรียบร้อยก่อน.")
        else:
            try:
                lstm_result = inference_lstm_model(
                    data=processed_df,
                    value_column=value_column,
                    days_to_remove=int(days_to_remove),
                    model_path=model_path,
                    scaler_path=scaler_path,
                    window_size=30  # ต้องตรงกับขณะที่เทรน
                )
                results['LSTM'] = lstm_result
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดใน LSTM: {e}")

    # ---------- ตรวจสอบว่ามีโมเดลใดบ้างที่รันสำเร็จ ----------
    if not results:
        st.error("ไม่พบผลลัพธ์จากโมเดลใดเลย")
        st.stop()

    # ---------- สร้างกราฟเปรียบเทียบ ----------
    st.subheader("เปรียบเทียบ Actual vs Predicted ของทุกโมเดล")

    fig = go.Figure()

    # 1) Plot เส้น Actual เต็มช่วง
    actual_series = processed_df[value_column]
    fig.add_trace(go.Scatter(
        x=actual_series.index,
        y=actual_series,
        mode='lines',
        name='Actual',
        line=dict(color='black')
    ))

    # 2) Plot เส้น Predicted ของแต่ละโมเดล
    for model_name, result_data in results.items():
        comp_df = result_data['comparison']
        fig.add_trace(go.Scatter(
            x=comp_df.index,
            y=comp_df['Predicted'],
            mode='lines',
            name=f'Predicted ({model_name})'
        ))

    fig.update_layout(
        title='Actual vs Predicted (All Models)',
        xaxis_title='Date',
        yaxis_title=value_column,
        legend=dict(x=0, y=1),
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------- สร้างตาราง Metrics เปรียบเทียบ ----------
    st.subheader("ตารางเปรียบเทียบ Metrics (MAE, RMSE, MAPE)")
    metrics_data = []
    for model_name, result_data in results.items():
        row = {
            'Model': model_name,
            'MAE': result_data['mae'],
            'RMSE': result_data['rmse'],
            'MAPE': result_data['mape']
        }
        metrics_data.append(row)

    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True)

    # ---------- แสดงตารางเปรียบเทียบ Actual vs Predicted (แบบเต็ม) ----------
    st.subheader("ตารางเปรียบเทียบ Actual vs Predicted แยกตามโมเดล")
    for model_name, result_data in results.items():
        st.markdown(f"**{model_name}**")
        comp_df = result_data['comparison'].dropna(subset=['Actual', 'Predicted'])
        st.dataframe(comp_df, use_container_width=True)

if __name__ == "__main__":
    main()