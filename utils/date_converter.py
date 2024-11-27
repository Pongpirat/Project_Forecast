# utils/date_converter.py

from datetime import datetime
import streamlit as st

def convert_thai_date_to_datetime(thai_date):
    months = {
        'ม.ค.': '01', 'ก.พ.': '02', 'มี.ค.': '03', 'เม.ย.': '04',
        'พ.ค.': '05', 'มิ.ย.': '06', 'ก.ค.': '07', 'ส.ค.': '08',
        'ก.ย.': '09', 'ต.ค.': '10', 'พ.ย.': '11', 'ธ.ค.': '12'
    }
    try:
        parts = thai_date.strip().split(' ')
        if len(parts) != 3:
            raise ValueError("รูปแบบวันที่ไม่ถูกต้อง")
        day, month, year = parts
        year = str(int(year) - 543)  # แปลงปี พ.ศ. เป็น ค.ศ.
        month = months.get(month, '01')  # แปลงเดือนภาษาไทยเป็นตัวเลข, กำหนดเป็น '01' หากไม่พบ
        return datetime.strptime(f"{day}/{month}/{year}", "%d/%m/%Y")
    except Exception as e:
        # แสดงข้อความข้อผิดพลาดและหยุดการทำงาน
        st.error(f"Error converting date '{thai_date}': {e}")
        st.stop()