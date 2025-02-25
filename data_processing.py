import pandas as pd
from utils.date_converter import convert_thai_date_to_datetime

def process_data(df, value_column):
    """
    ประมวลผลข้อมูล:
    - ล้างชื่อคอลัมน์ (กัน BOM, เว้นวรรคเกิน)
    - แปลงวันที่ (ไทย -> สากล) โดยอ้างอิงคอลัมน์ 'งวด'
    - set_index('งวด')
    - เติมวันที่ที่ขาด (reindex daily) จากปี 2014-01-01 จนถึงวันสุดท้าย
    - forward fill
    - เก็บค่าเดิมในคอลัมน์ raw_value
    - ปัดเศษตัวเลข
    - เพิ่มคอลัมน์ day, month, year, week
    """

    df.columns = df.columns.str.strip().str.replace('\ufeff', '', regex=True)
    if 'งวด' not in df.columns:
        if df.index.name == 'งวด':
            df.reset_index(inplace=True)
        else:
            raise ValueError("ไม่พบคอลัมน์ 'งวด' ใน DataFrame")
    df['งวด'] = df['งวด'].apply(convert_thai_date_to_datetime)
    df.dropna(subset=['งวด'], inplace=True)
    df.set_index('งวด', inplace=True)
    df.sort_index(inplace=True)

    numeric_columns = df.select_dtypes(include=['float', 'int']).columns
    if value_column not in numeric_columns:
        raise ValueError(f"ไม่พบคอลัมน์ '{value_column}' ในข้อมูลที่ผ่านการประมวลผล")

    start_date = max(pd.to_datetime('2014-01-01'), df.index.min())
    end_date = df.index.max()
    full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    df = df.reindex(full_date_range)
    df.index.name = 'งวด'
    df.fillna(method='ffill', inplace=True)

    df['raw_value'] = df[value_column].copy()
    df[numeric_columns] = df[numeric_columns].round(4)
    df.dropna(inplace=True)

    df['day'] = df.index.day
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['week'] = df.index.isocalendar().week

    numeric_columns = df.select_dtypes(include=['float', 'int', 'uint32']).columns

    return df, numeric_columns
