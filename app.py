import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go

from data_processing import process_data
from models.exponential_smoothing import exponential_smoothing_model
from models.arima_model import arima_model  # เพิ่มการนำเข้าโมเดล ARIMA
from utils.date_converter import convert_thai_date_to_datetime

# ตั้งค่าหน้าเว็บและหัวข้อหลัก
st.set_page_config(page_title="ประมวลผลข้อมูลอัตราแลกเปลี่ยน", layout="wide")
st.title("การพยากรณ์อัตราแลกเปลี่ยนเงินสกุลบาทไทยเป็น 10 สกุลเงินต่างประเทศที่สำคัญในตลาดส่งออก")
st.markdown("---")

# Sidebar สำหรับเลือกไฟล์และเลือกโมเดล
with st.sidebar:
    st.header("การตั้งค่า")

    # กำหนดโฟลเดอร์ที่มีไฟล์ล่วงหน้า
    preloaded_files_dir = os.path.join(os.getcwd(), 'currency')  # ใช้โฟลเดอร์ใน Working Directory
    if not os.path.exists(preloaded_files_dir):
        st.error(f"ไม่พบโฟลเดอร์ '{preloaded_files_dir}'. กรุณาสร้างและเพิ่มไฟล์ CSV ที่ต้องการ.")
        st.stop()
    
    preloaded_files = [f for f in os.listdir(preloaded_files_dir) if f.endswith('.csv')]
    if not preloaded_files:
        st.error(f"ไม่พบไฟล์ CSV ในโฟลเดอร์ '{preloaded_files_dir}'.")
        st.stop()

    selected_preloaded_file = st.selectbox("เลือกไฟล์จากไฟล์ที่มีอยู่แล้ว", preloaded_files)
    preloaded_file_path = os.path.join(preloaded_files_dir, selected_preloaded_file)

    # เลือกโมเดลสำหรับการทำนาย
    st.subheader("เลือกโมเดลสำหรับการทำนาย")
    model_selection = st.radio(
        "กรุณาเลือกโมเดลที่ต้องการใช้งาน:",
        ('Exponential Smoothing', 'ARIMA', 'Moving Average'),  # เพิ่ม ARIMA เป็นตัวเลือก
        index=0
    )

    # หากเลือก Exponential Smoothing ให้แสดงตัวเลือกเพิ่มเติม
    if model_selection == 'Exponential Smoothing':
        st.subheader("ตั้งค่าสำหรับ Exponential Smoothing")
        days_to_remove = st.number_input("ระบุจำนวนวันที่ต้องการแยกสำหรับการทดสอบ", min_value=1, value=30)
    elif model_selection == 'ARIMA':
        st.subheader("ตั้งค่าสำหรับ ARIMA")
        days_to_remove = st.number_input("ระบุจำนวนวันที่ต้องการแยกสำหรับการทดสอบ", min_value=1, value=30)
        order_p = st.number_input("ระบุค่า p (AR)", min_value=0, value=1)
        order_d = st.number_input("ระบุค่า d (Differencing)", min_value=0, value=1)
        order_q = st.number_input("ระบุค่า q (MA)", min_value=0, value=1)
        arima_order = (order_p, order_d, order_q)
    else:
        days_to_remove = None

# โหลดและตรวจสอบไฟล์ที่เลือก
try:
    # พยายามอ่านไฟล์ที่มีไว้ล่วงหน้าด้วย encoding 'utf-8'
    df = pd.read_csv(preloaded_file_path, encoding='utf-8')
except UnicodeDecodeError:
    # หากอ่านด้วย utf-8 ไม่ได้ ลองใช้ 'cp874'
    try:
        df = pd.read_csv(preloaded_file_path, encoding='cp874')
    except Exception as e:
        st.error(f"ไม่สามารถอ่านไฟล์ได้: {e}")
        st.stop()

# ตรวจสอบว่ามีคอลัมน์ 'งวด' หรือไม่
if 'งวด' not in df.columns:
    st.error("ไม่พบคอลัมน์ 'งวด' ในไฟล์ CSV ที่เลือก")
    st.stop()

# ประมวลผลข้อมูล
with st.spinner('กำลังประมวลผลข้อมูล...'):
    processed_df, numeric_cols = process_data(df)

# ย้ายการเลือกคอลัมน์ค่าที่ต้องการทำนายไปที่ Sidebar
value_column = None
if model_selection in ['Exponential Smoothing', 'ARIMA'] and days_to_remove is not None:
    # เพิ่ม selectbox ใน Sidebar
    value_column = st.sidebar.selectbox(
        "เลือกคอลัมน์ค่าที่ต้องการทำนาย",
        options=numeric_cols
    )

# หากเลือกโมเดลให้ดำเนินการต่อ
if model_selection == 'Exponential Smoothing':
    if days_to_remove is not None and value_column:
        try:
            # เรียกใช้โมเดล Exponential Smoothing
            result = exponential_smoothing_model(
                data=processed_df,
                value_column=value_column,
                days_to_remove=int(days_to_remove)
            )

            # แสดงผลลัพธ์
            st.subheader("ผลการทำนาย")
            fig = go.Figure()

            # แสดง Actual
            fig.add_trace(go.Scatter(
                x=result['comparison'].index,
                y=result['comparison']['Actual'],
                mode='lines',
                name='Actual'
            ))

            # แสดง Predicted เฉพาะช่วงทำนาย
            fig.add_trace(go.Scatter(
                x=result['comparison'].index,
                y=result['comparison']['Predicted'],
                mode='lines',
                name='Predicted'
            ))

            # ปรับแต่งกราฟ
            fig.update_layout(
                title='Actual vs Predicted Exchange Rate',
                xaxis_title='Date',
                yaxis_title='Exchange Rate',
                legend=dict(x=0, y=1),
                hovermode='x unified'
            )

            # สร้างแถวคอลัมน์สำหรับกราฟและตาราง
            col1, col2 = st.columns([2, 1])  # สามารถปรับขนาดคอลัมน์ได้ตามต้องการ

            with col1:
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # กรองข้อมูลให้แสดงเฉพาะแถวที่มีค่าจริงและค่าทำนาย
                filtered_comparison = result['comparison'].dropna(subset=['Actual', 'Predicted'])

                # เพิ่ม scroll ให้ตาราง
                st.subheader("ตารางเปรียบเทียบ Actual vs Predicted")
                st.dataframe(filtered_comparison, height=300, use_container_width=True)

            # แสดงค่า Error Metrics
            st.header("ผลค่าความแม่นยำในช่วงที่ลบข้อมูล", divider='gray')
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="**Mean Absolute Error (MAE)**", value=f"{result['mae']:.4f}")
            with col2:
                st.metric(label="**Root Mean Square Error (RMSE)**", value=f"{result['rmse']:.4f}")
            with col3:
                st.metric(label="**Mean Absolute Percentage Error (MAPE)**", value=f"{result['mape']:.2f}%")

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาด: {e}")
    else:
        st.warning("กรุณาระบุจำนวนวันที่ต้องการแยกสำหรับการทดสอบและเลือกคอลัมน์ค่าที่ต้องการทำนาย")

elif model_selection == 'ARIMA':
    if days_to_remove is not None and value_column:
        try:
            # เรียกใช้โมเดล ARIMA
            result = arima_model(
                data=processed_df,
                value_column=value_column,
                days_to_remove=int(days_to_remove),
                order=arima_order
            )

            # แสดงผลลัพธ์
            st.subheader("ผลการพยากรณ์ด้วย ARIMA")
            
            # แสดงกราฟ Original Data
            col1, col2, col3 = st.columns(3)
            with col1:
                st.plotly_chart(result['fig_original'], use_container_width=True)
            with col2:
                st.plotly_chart(result['fig_forecast'], use_container_width=True)
            with col3:
                st.plotly_chart(result['fig_combined'], use_container_width=True)

            # แสดงค่า Accuracy Metrics
            st.header("ผลค่าความแม่นยำในช่วงที่ลบข้อมูล", divider='gray')
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="**Mean Absolute Error (MAE)**", value=f"{result['mae']:.4f}")
            with col2:
                st.metric(label="**Root Mean Square Error (RMSE)**", value=f"{result['rmse']:.4f}")
            with col3:
                st.metric(label="**Mean Absolute Percentage Error (MAPE)**", value=f"{result['mape']:.2f}%")

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาด: {e}")
    else:
        st.warning("กรุณาระบุจำนวนวันที่ต้องการแยกสำหรับการทดสอบและเลือกคอลัมน์ค่าที่ต้องการทำนาย")

elif model_selection == 'Moving Average':
    st.subheader("Moving Average for Exchange Rate Prediction")
    st.info("กำลังดำเนินการเพิ่มเติมสำหรับโมเดล Moving Average...")
    st.write("รอการดำเนินการเพิ่มเติมจากคุณ...")