import pandas as pd
import streamlit as st
from utils.date_converter import convert_thai_date_to_datetime

def process_data(df):
    try:
        # แปลงวันที่
        df['งวด'] = df['งวด'].apply(convert_thai_date_to_datetime)
    except ValueError as e:
        st.error(f"การแปลงวันที่ล้มเหลว: {e}")
        st.stop()  # หยุดการทำงานของ Streamlit

    # ลบแถวที่วันที่ไม่สามารถแปลงได้
    df = df.dropna(subset=['งวด'])

    # ตั้งค่า "งวด" เป็น index
    df.set_index('งวด', inplace=True)

    # ตรวจสอบชนิดข้อมูลของคอลัมน์อัตราแลกเปลี่ยน
    numeric_columns = df.select_dtypes(include=['float', 'int']).columns

    # สร้างช่วงวันที่ใหม่ เริ่มต้นจาก 1/1/2014 ถึงวันที่สูงสุดในข้อมูล
    start_date = pd.to_datetime('2014-01-01')
    end_date = df.index.max()
    full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # เติมวันที่ขาดหายไป
    df = df.reindex(full_date_range)
    df.index.name = 'งวด'

    # ใช้ interpolation เพื่อเติมข้อมูลที่ขาด
    df.interpolate(method='linear', inplace=True)

    # เติมค่า NaN ที่ยังขาดด้วยการ back-fill (เติมค่าจากวันที่ถัดไป)
    df.fillna(method='bfill', inplace=True)

    # ปัดเศษค่าทศนิยมให้ตรงกับข้อมูลต้นฉบับ (สมมติว่ามี 4 ตำแหน่งทศนิยม)
    decimal_places = 4
    df[numeric_columns] = df[numeric_columns].round(decimal_places)

    return df, numeric_columns