import pandas as pd
from utils.date_converter import convert_thai_date_to_datetime

def process_data(df, value_column):
    try:
        # แปลงวันที่
        df['งวด'] = df['งวด'].apply(convert_thai_date_to_datetime)
    except ValueError as e:
        print(f"Error converting date: {e}")
        raise e

    # ลบแถวที่ไม่สามารถแปลงวันที่ได้
    df.dropna(subset=['งวด'], inplace=True)

    # ตั้ง "งวด" เป็น index
    df.set_index('งวด', inplace=True)

    # ตรวจสอบคอลัมน์ที่เป็นตัวเลข
    numeric_columns = df.select_dtypes(include=['float', 'int']).columns
    if value_column not in numeric_columns:
        raise ValueError(f"ไม่พบคอลัมน์ '{value_column}' ในข้อมูลที่ผ่านการประมวลผล")

    # สร้างช่วงวันที่ใหม่
    start_date = pd.to_datetime('2014-01-01')
    end_date = df.index.max()
    full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # เติมวันที่ขาด
    df = df.reindex(full_date_range)
    df.index.name = 'งวด'

    # forward-fill และ back-fill
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    # ปัดเศษตัวเลข
    decimal_places = 4
    df[numeric_columns] = df[numeric_columns].round(decimal_places)

    # ----- แทนที่การสร้าง lag_ เป็นการสร้าง Smooth Feature -----
    # เช่น ใช้ Moving Average 7 วัน และ 14 วัน แบบง่าย
    df['smooth_7'] = df[value_column].rolling(window=7).mean()
    df['smooth_14'] = df[value_column].rolling(window=14).mean()

    # ลบ NaN หลังจากสร้าง Smooth (กรณีต้นข้อมูล)
    df.dropna(inplace=True)

    # อัปเดตคอลัมน์ตัวเลข
    numeric_columns = df.select_dtypes(include=['float', 'int']).columns

    return df, numeric_columns