import pandas as pd
from utils.date_converter import convert_thai_date_to_datetime

import pandas as pd
from utils.date_converter import convert_thai_date_to_datetime

def process_data(df, value_column, smoothing_window=3):
    """
    ประมวลผลข้อมูล:
    - แปลงวันที่ (ไทย -> สากล)
    - เติมวันที่ที่ขาด (reindex daily) จากปี 2014-01-01 จนถึงวันสุดท้าย
    - forward fill เพื่อเลี่ยง data leakage
    - เก็บค่าเดิมในคอลัมน์ raw_value
    - ทำ rolling average (window=smoothing_window) -> smoothed_value
    - ปัดเศษตัวเลข
    - เพิ่มคอลัมน์ day, month, year, week
    """

    # 1) แปลงวันที่
    try:
        df['งวด'] = df['งวด'].apply(convert_thai_date_to_datetime)
    except ValueError as e:
        print(f"Error converting date: {e}")
        raise e

    df.dropna(subset=['งวด'], inplace=True)
    df.set_index('งวด', inplace=True)
    df.sort_index(inplace=True)

    # 2) กรองคอลัมน์ตัวเลข
    numeric_columns = df.select_dtypes(include=['float', 'int']).columns
    if value_column not in numeric_columns:
        raise ValueError(f"ไม่พบคอลัมน์ '{value_column}' ในข้อมูลที่ผ่านการประมวลผล")

    # 3) reindex ให้เป็นรายวัน (Daily)
    start_date = max(pd.to_datetime('2014-01-01'), df.index.min())
    end_date = df.index.max()
    full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    df = df.reindex(full_date_range)
    df.index.name = 'งวด'
    df.fillna(method='ffill', inplace=True)

    # 4) เก็บค่า "อัตราขาย" เดิมไว้เป็น raw_value
    df['raw_value'] = df[value_column].copy()

    # 5) สร้าง smoothed_value จาก rolling
    df['smoothed_value'] = df[value_column].rolling(window=smoothing_window, min_periods=1).mean()

    # ปัดเศษตัวเลขทุกคอลัมน์ตัวเลข
    df[numeric_columns] = df[numeric_columns].round(4)

    # 6) ลบแถวที่อาจยังมี NaN ใน smoothed_value (ถ้ามี)
    df.dropna(subset=['smoothed_value'], inplace=True)

    # 7) เพิ่มคอลัมน์วันที่
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['week'] = df.index.isocalendar().week

    # อัปเดต numeric_columns ให้รวม raw_value และ smoothed_value ด้วย
    numeric_columns = df.select_dtypes(include=['float', 'int', 'uint32']).columns

    return df, numeric_columns
