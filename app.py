import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# import ฟังก์ชันจากโมดูลต่าง ๆ
from data_processing import process_data
from models.exponential_smoothing import exponential_smoothing_model
from models.moving_average import moving_average_model
from models.sarima_model import sarima_model
from models.lstm_inference import inference_lstm_model
from train_lstm import train_lstm_model

# (ลบ @st.cache_data ในไฟล์โมเดล FULL)
from models.exponential_smoothing_full import exponential_smoothing_full_model
from models.moving_average_full import moving_average_full_model
from models.sarima_model_full import sarima_full_model
from models.lstm_inference_full import inference_lstm_full_model


########################################
# ฟังก์ชันช่วยแสดงผล
########################################
def render_historical_metric_card(title: str, mae, rmse, mape):
    if isinstance(mae, (float, int)):
        mae_str = f"{mae:.2f}"
    else:
        mae_str = str(mae)

    if isinstance(rmse, (float, int)):
        rmse_str = f"{rmse:.2f}"
    else:
        rmse_str = str(rmse)

    if isinstance(mape, (float, int)):
        mape_str = f"{mape:.2f}%"
    else:
        mape_str = str(mape)

    card_html = f"""
    <div style="
        border: 1px solid #ccc;
        border-radius: 10px;
        padding: 15px;
        margin: 15px 0;
        min-height: 150px;
        text-align: center;">
      <div style="margin: 0; color: #4d4d4d; font-size: 1.0rem; font-weight: bold;">
        {title}
      </div>
      <p style="margin: 0px 0 0 0; font-size: 1.8rem; font-weight: bold; color: #4d4d4d;">
        MAPE: {mape_str}
      </p>
      <p style="margin: 0; font-size: 1.0rem; color: #28a745;">
        MAE: {mae_str}, RMSE: {rmse_str}
      </p>
    </div>
    """
    return card_html

def get_historical_metrics_data(results, actual_series):
    metrics_data = []
    for model_name, result_data in results.items():
        if (not result_data) or ('comparison' not in result_data):
            metrics_data.append({
                'Model': model_name,
                'MAE': '-',
                'RMSE': '-',
                'MAPE': '-'
            })
        else:
            comp_df = result_data['comparison']
            common_index = actual_series.dropna().index.intersection(comp_df['Predicted'].dropna().index)
            if common_index.empty:
                metrics_data.append({
                    'Model': model_name,
                    'MAE': '-',
                    'RMSE': '-',
                    'MAPE': '-'
                })
            else:
                actual_aligned = actual_series.loc[common_index]
                predicted_aligned = comp_df.loc[common_index, 'Predicted']
                mae = mean_absolute_error(actual_aligned, predicted_aligned)
                rmse = np.sqrt(mean_squared_error(actual_aligned, predicted_aligned))
                mape = np.mean(np.abs((actual_aligned - predicted_aligned) / actual_aligned)) * 100
                metrics_data.append({
                    'Model': model_name,
                    'MAE': mae,
                    'RMSE': rmse,
                    'MAPE': mape
                })
    return metrics_data

def find_metric_row(metrics_data_list, model_name: str):
    for row in metrics_data_list:
        if row['Model'] == model_name:
            return row
    return None


########################################
# ฟังก์ชันโหลดไฟล์ (พร้อม cache)
########################################
@st.cache_data
def load_csv_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='cp874')
    return df


########################################
# ฟังก์ชัน Train/Run โมเดล Historical
########################################
def run_all_models(processed_df: pd.DataFrame, value_column: str, days_to_remove: int,
                   forecast_days: int, model_path: str, scaler_date_path: str, scaler_target_path: str):
    results = {}
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
        print(f"[ERROR] Moving Average (EMA): {e}")

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

    # LSTM (Historical)
    if os.path.exists(model_path) and os.path.exists(scaler_date_path) and os.path.exists(scaler_target_path):
        try:
            lstm_result = inference_lstm_model(
                data=processed_df,
                value_column=value_column,
                days_to_remove=int(days_to_remove),
                model_path=model_path,
                scaler_date_path=scaler_date_path,
                scaler_target_path=scaler_target_path,
                window_size=14,
                forecast_days=int(forecast_days)  # ในที่นี้ set=0
            )
            results['LSTM'] = lstm_result
        except Exception as e:
            results['LSTM'] = None
            print(f"[ERROR] LSTM: {e}")
    else:
        results['LSTM'] = None

    return results


########################################
# ฟังก์ชัน Single Pipeline (Cache ตรงนี้ที่เดียว)
########################################
@st.cache_data
def full_pipeline(
    hist_file_path: str,
    out_file_path: str,
    base_filename: str,
    days_to_remove: int,
    forecast_days: int
):
    """
    รวมทุกขั้นตอน: โหลดไฟล์, process, รันโมเดล Historical, รันโมเดล Full Data
    แล้ว return ผลลัพธ์ทั้งหมดในรูป dict
    """

    # --------------------------
    # 1) โหลดไฟล์ Historical
    # --------------------------
    df_hist = load_csv_data(hist_file_path)
    processed_hist_df, _ = process_data(df_hist, 'อัตราขาย')

    # --------------------------
    # 2) โหลดไฟล์ 30 วัน (ถ้ามี)
    # --------------------------
    if out_file_path is not None and os.path.exists(out_file_path):
        df_out = load_csv_data(out_file_path)
        processed_out_df, _ = process_data(df_out, 'อัตราขาย')
    else:
        processed_out_df = None

    # --------------------------
    # 3) รันโมเดล Historical
    # --------------------------
    model_folder = os.path.join('models', base_filename)
    model_path = os.path.join(model_folder, 'lstm_model.h5')
    scaler_date_path = os.path.join(model_folder, 'scaler_lstm_date.pkl')
    scaler_target_path = os.path.join(model_folder, 'scaler_lstm_target.pkl')

    historical_results = run_all_models(
        processed_df=processed_hist_df,
        value_column='อัตราขาย',
        days_to_remove=days_to_remove,
        forecast_days=0,  # ไม่ Forecast ใน Historical
        model_path=model_path,
        scaler_date_path=scaler_date_path,
        scaler_target_path=scaler_target_path
    )

    # --------------------------
    # 4) รวม Actual ทั้งหมด
    # --------------------------
    if processed_out_df is not None:
        combined_actual = pd.concat([processed_hist_df['raw_value'], processed_out_df['raw_value']])
    else:
        combined_actual = processed_hist_df['raw_value']

    # --------------------------
    # 5) รันโมเดล Full Data
    # --------------------------
    # (Exponential Smoothing Full)
    try:
        forecast_es = exponential_smoothing_full_model(processed_hist_df, 'อัตราขาย', 30)
    except Exception as e:
        forecast_es = None
        print(f"[ERROR] Exponential Smoothing (Full Data): {e}")

    # (Moving Average Full)
    try:
        forecast_ma = moving_average_full_model(processed_hist_df, 'อัตราขาย', 30, window_size=30, use_ema=True)
    except Exception as e:
        forecast_ma = None
        print(f"[ERROR] Moving Average (Full Data): {e}")

    # (SARIMA Full)
    try:
        forecast_sarima = sarima_full_model(processed_hist_df, 'อัตราขาย', 30,
                                            smooth_window=3,
                                            seasonal_order=(1, 1, 1, 7),
                                            order=(2, 1, 2))
    except Exception as e:
        forecast_sarima = None
        print(f"[ERROR] SARIMA (Full Data): {e}")

    # (LSTM Full)
    lstm_full_result_df = None
    full_model_folder = os.path.join('models', base_filename, 'full_data')
    full_model_path = os.path.join(full_model_folder, 'lstm_model_full.h5')
    full_scaler_date_path = os.path.join(full_model_folder, 'scaler_lstm_date_full.pkl')
    full_scaler_target_path = os.path.join(full_model_folder, 'scaler_lstm_target_full.pkl')
    if (os.path.exists(full_model_path)
        and os.path.exists(full_scaler_date_path)
        and os.path.exists(full_scaler_target_path)):
        try:
            lstm_full_result_df = inference_lstm_full_model(
                data=processed_hist_df,
                value_column='อัตราขาย',
                model_path=full_model_path,
                scaler_date_path=full_scaler_date_path,
                scaler_target_path=full_scaler_target_path,
                window_size=14,
                forecast_days=30
            )
        except Exception as e:
            lstm_full_result_df = None
            print(f"[ERROR] LSTM (Full Data): {e}")
    else:
        print("[INFO] Full LSTM Model Not Found")

    # --------------------------
    # 6) คืนทุกอย่างเป็น dict
    # --------------------------
    return {
        "processed_hist_df": processed_hist_df,
        "processed_out_df": processed_out_df,
        "combined_actual": combined_actual,  # สำหรับ plot/filter slider
        "results_hist": historical_results,   # Exponential / MA / SARIMA / LSTM (Historical)
        "forecast_es": forecast_es,          # Full Data
        "forecast_ma": forecast_ma,
        "forecast_sarima": forecast_sarima,
        "forecast_lstm": lstm_full_result_df
    }


########################################
# ส่วน main
########################################
def main():
    st.set_page_config(page_title="Compare All Models", layout="wide")
    st.title("Dashboard เปรียบเทียบการพยากรณ์จากทุกโมเดลในหน้าเดียว")
    st.markdown("---")

    # --------------------------
    # Sidebar: เลือกไฟล์
    # --------------------------
    with st.sidebar:
        st.header("การตั้งค่า")
        preloaded_files_dir = os.path.join(os.getcwd(), 'currency')
        if not os.path.exists(preloaded_files_dir):
            st.error(f"ไม่พบโฟลเดอร์ '{preloaded_files_dir}'")
            st.stop()
        historical_files = [f for f in os.listdir(preloaded_files_dir) if f.endswith('.csv')]
        if not historical_files:
            st.error(f"ไม่พบไฟล์ CSV ในโฟลเดอร์ '{preloaded_files_dir}'")
            st.stop()

        # เลือกไฟล์
        selected_hist_file = st.selectbox("เลือกไฟล์ Historical (Training)", historical_files)
        hist_file_path = os.path.join(preloaded_files_dir, selected_hist_file)
        base_filename = os.path.splitext(selected_hist_file)[0]

        # เลือกไฟล์ 30 วัน (ถ้ามี)
        currency30_dir = os.path.join(os.getcwd(), 'currency30')
        currency30_files = []
        if os.path.exists(currency30_dir):
            for f in os.listdir(currency30_dir):
                if f.endswith('.csv') and base_filename in f:
                    currency30_files.append(f)
        if currency30_files:
            if len(currency30_files) == 1:
                out_file_path = os.path.join(currency30_dir, currency30_files[0])
            else:
                selected_out_file = st.selectbox(f"เลือกไฟล์ข้อมูลเพิ่มเติม (30 วัน) สำหรับ {base_filename}", currency30_files)
                out_file_path = os.path.join(currency30_dir, selected_out_file)
        else:
            st.info(f"ไม่พบไฟล์ข้อมูลเพิ่มเติม (30 วัน) สำหรับ {base_filename} ในโฟลเดอร์ 'currency30'")
            out_file_path = None

        days_to_remove = st.number_input("ระบุจำนวนวันที่ต้องการแยกทดสอบ (Test Set)", min_value=1, value=30)
        forecast_days = 30  # สมมติว่าใช้ 30 วัน
        st.markdown("---")

        # LSTM (Historical) ถ้าไม่มีไฟล์โมเดล ก็ให้ปุ่ม Train
        model_folder = os.path.join('models', base_filename)
        model_path = os.path.join(model_folder, 'lstm_model.h5')
        scaler_date_path = os.path.join(model_folder, 'scaler_lstm_date.pkl')
        scaler_target_path = os.path.join(model_folder, 'scaler_lstm_target.pkl')
        if not (os.path.exists(model_path) and os.path.exists(scaler_date_path) and os.path.exists(scaler_target_path)):
            st.warning(f"ไม่พบโมเดล LSTM หรือ Scaler สำหรับไฟล์ '{selected_hist_file}'")
            if st.button("ฝึกโมเดล LSTM สำหรับไฟล์นี้"):
                train_success = train_lstm_model(selected_hist_file)
                if train_success:
                    st.success(f"ฝึกโมเดล LSTM สำเร็จสำหรับไฟล์ '{selected_hist_file}'")
                else:
                    st.error(f"การฝึกโมเดล LSTM ล้มเหลวสำหรับไฟล์ '{selected_hist_file}'")

    # --------------------------
    # เรียกฟังก์ชัน Pipeline
    # --------------------------
    with st.spinner('กำลังประมวลผล + รันโมเดลทั้งหมด...'):
        data_dict = full_pipeline(
            hist_file_path=hist_file_path,
            out_file_path=out_file_path,
            base_filename=base_filename,
            days_to_remove=days_to_remove,
            forecast_days=forecast_days
        )

    # ดึงผลลัพธ์ออกมา
    processed_hist_df = data_dict["processed_hist_df"]
    processed_out_df  = data_dict["processed_out_df"]
    combined_actual   = data_dict["combined_actual"]
    results_hist      = data_dict["results_hist"]   # Historical
    forecast_es       = data_dict["forecast_es"]
    forecast_ma       = data_dict["forecast_ma"]
    forecast_sarima   = data_dict["forecast_sarima"]
    lstm_forecast_df  = data_dict["forecast_lstm"]  # LSTM Full Data

    # --------------------------
    # เลือกช่วงเดือนสำหรับ filter
    # --------------------------
    max_date = combined_actual.index.max().date()
    min_date = combined_actual.index.min().date()
    default_start = (combined_actual.index.max() - pd.DateOffset(months=9)).date()

    with st.sidebar:
        selected_date_range = st.slider(
            "เลือกช่วงเดือนที่ต้องการแสดง",
            min_value=min_date,
            max_value=max_date,
            value=(default_start, max_date),
            format="MM/YYYY"
        )

    def filter_by_date_range(series, date_range):
        start, end = date_range
        return series[(series.index.date >= start) & (series.index.date <= end)]

    # --------------------------
    # Plot
    # --------------------------
    col_left, col_right = st.columns([3,1])

    with col_left:
        fig = go.Figure()
        filtered_actual = filter_by_date_range(combined_actual, selected_date_range)
        fig.add_trace(go.Scatter(
            x=filtered_actual.index,
            y=filtered_actual,
            mode='lines',
            name='Actual (Raw)',
            line=dict(color='black')
        ))

        # Historical
        if results_hist:
            for model_name, result_data in results_hist.items():
                if not result_data:
                    continue
                comp_df = result_data['comparison']
                comp_filtered = comp_df[(comp_df.index.date >= selected_date_range[0]) &
                                        (comp_df.index.date <= selected_date_range[1])]
                fig.add_trace(go.Scatter(
                    x=comp_filtered.index,
                    y=comp_filtered['Predicted'],
                    mode='lines',
                    name=f'{model_name}'
                ))

        # Full Data
        if forecast_es and 'comparison' in forecast_es:
            es_df = forecast_es['comparison']
            fig.add_trace(go.Scatter(
                x=es_df.index,
                y=es_df['Predicted'],
                mode='lines',
                name='Exp. Smoothing (Full Data)',
                line=dict(dash='dash', color='#86caff')
            ))
        if forecast_ma and 'comparison' in forecast_ma:
            ma_df = forecast_ma['comparison']
            fig.add_trace(go.Scatter(
                x=ma_df.index,
                y=ma_df['Predicted'],
                mode='lines',
                name='Moving Average (Full Data)',
                line=dict(dash='dash', color='#ff3131')
            ))
        if forecast_sarima and 'comparison' in forecast_sarima:
            sarima_df = forecast_sarima['comparison']
            fig.add_trace(go.Scatter(
                x=sarima_df.index,
                y=sarima_df['Predicted'],
                mode='lines',
                name='SARIMA (Full Data)',
                line=dict(dash='dash', color='#ffadad')
            ))
        if lstm_forecast_df is not None:
            fig.add_trace(go.Scatter(
                x=lstm_forecast_df.index,
                y=lstm_forecast_df['Predicted'],
                mode='lines',
                name='LSTM (Full Data)',
                line=dict(dash='dash', color='#2db19f')
            ))

        fig.update_layout(
            title=f'Actual vs Historical vs Full Data Model - {selected_date_range[0].strftime("%m/%Y")} ถึง {selected_date_range[1].strftime("%m/%Y")}',
            xaxis_title='Date',
            yaxis_title='อัตราขาย',
            legend=dict(x=0, y=1),
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

    # --------------------------------
    # Metrics (Historical Models)
    # --------------------------------
    with col_right:
        st.subheader("Metrics (Historical Models)")
        hist_metrics_data = get_historical_metrics_data(results_hist, processed_hist_df['raw_value'])
        row_exp   = find_metric_row(hist_metrics_data, "Exponential Smoothing")
        row_ma    = find_metric_row(hist_metrics_data, "Moving Average (EMA)")
        row_sarim = find_metric_row(hist_metrics_data, "SARIMA")
        row_lstm  = find_metric_row(hist_metrics_data, "LSTM")
        st.markdown(render_historical_metric_card(
            "Exponential Smoothing",
            row_exp['MAE'] if row_exp else "-",
            row_exp['RMSE'] if row_exp else "-",
            row_exp['MAPE'] if row_exp else "-"
        ), unsafe_allow_html=True)
        st.markdown(render_historical_metric_card(
            "Moving Average (EMA)",
            row_ma['MAE'] if row_ma else "-",
            row_ma['RMSE'] if row_ma else "-",
            row_ma['MAPE'] if row_ma else "-"
        ), unsafe_allow_html=True)
        st.markdown(render_historical_metric_card(
            "SARIMA",
            row_sarim['MAE'] if row_sarim else "-",
            row_sarim['RMSE'] if row_sarim else "-",
            row_sarim['MAPE'] if row_sarim else "-"
        ), unsafe_allow_html=True)
        st.markdown(render_historical_metric_card(
            "LSTM",
            row_lstm['MAE'] if row_lstm else "-",
            row_lstm['RMSE'] if row_lstm else "-",
            row_lstm['MAPE'] if row_lstm else "-"
        ), unsafe_allow_html=True)

    # --------------------------------
    # ตารางเปรียบเทียบ (Historical)
    # --------------------------------
    with col_left:
        st.subheader("ตารางเปรียบเทียบ Actual vs Predicted (Historical)")
        combined_df = pd.DataFrame(index=processed_hist_df.index)
        combined_df['Actual (Raw)'] = processed_hist_df['raw_value']
        if results_hist:
            for model_name, result_data in results_hist.items():
                if not result_data:
                    continue
                comp_df = result_data['comparison']
                predicted_col_name = f'Predicted_{model_name}'
                combined_df = combined_df.join(comp_df['Predicted'].rename(predicted_col_name), how='left')

        # filter เฉพาะแถวที่ไม่มี NaN
        non_nan_columns = ['Actual (Raw)'] + [f'Predicted_{m}' for m in results_hist.keys() if results_hist[m] is not None]
        filtered_combined_df = combined_df.dropna(subset=non_nan_columns)
        filtered_combined_df = filtered_combined_df[
            (filtered_combined_df.index.date >= selected_date_range[0]) &
            (filtered_combined_df.index.date <= selected_date_range[1])
        ]
        st.dataframe(filtered_combined_df, use_container_width=True, height=240)
    st.markdown("---")

    # --------------------------------
    # ตารางทำนาย 30 วัน (Full Data)
    # --------------------------------
    if lstm_forecast_df is not None:
        combined_forecast = lstm_forecast_df.copy()
        if processed_out_df is not None:
            combined_forecast = combined_forecast.join(processed_out_df['raw_value'].rename("Actual (Raw)"), how='left')
        else:
            combined_forecast["Actual (Raw)"] = None
        combined_forecast = combined_forecast.rename(columns={'Predicted': 'LSTM (Full Data)'})

        if (forecast_es is not None) and ("comparison" in forecast_es):
            combined_forecast = combined_forecast.join(
                forecast_es['comparison'].rename(columns={'Predicted': 'Exp. Smoothing (Full Data)'}),
                how='left'
            )
        if (forecast_ma is not None) and ("comparison" in forecast_ma):
            combined_forecast = combined_forecast.join(
                forecast_ma['comparison'].rename(columns={'Predicted': 'Moving Average (Full Data)'}),
                how='left'
            )
        if (forecast_sarima is not None) and ("comparison" in forecast_sarima):
            combined_forecast = combined_forecast.join(
                forecast_sarima['comparison'].rename(columns={'Predicted': 'SARIMA (Full Data)'}),
                how='left'
            )

        # จัดลำดับคอลัมน์
        cols = combined_forecast.columns.tolist()
        if "Actual (Raw)" in cols:
            cols = ["Actual (Raw)"] + [c for c in cols if c != "Actual (Raw)"]
            combined_forecast = combined_forecast[cols]

        st.subheader("ตารางทำนาย 30 วันข้างหน้า (Full Data Model)")
        st.dataframe(combined_forecast, use_container_width=True)
    else:
        st.info("ไม่มีข้อมูล Forecast จาก Full Data Model")

    # --------------------------------
    # Metrics (Full Data) ถ้ามี Out
    # --------------------------------
    full_metrics = {}
    if processed_out_df is not None:
        actual_forecast = processed_out_df['raw_value']

        # Exp. Smoothing (Full)
        if forecast_es and 'comparison' in forecast_es:
            es_df = forecast_es['comparison']
            es_pred = es_df.loc[actual_forecast.index, 'Predicted']
            mae = mean_absolute_error(actual_forecast, es_pred)
            rmse = np.sqrt(mean_squared_error(actual_forecast, es_pred))
            mape = np.mean(np.abs((actual_forecast - es_pred) / actual_forecast)) * 100
            full_metrics['Exp. Smoothing Full'] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

        # Moving Average (Full)
        if forecast_ma and 'comparison' in forecast_ma:
            ma_df = forecast_ma['comparison']
            ma_pred = ma_df.loc[actual_forecast.index, 'Predicted']
            mae = mean_absolute_error(actual_forecast, ma_pred)
            rmse = np.sqrt(mean_squared_error(actual_forecast, ma_pred))
            mape = np.mean(np.abs((actual_forecast - ma_pred) / actual_forecast)) * 100
            full_metrics['Moving Average Full'] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

        # SARIMA (Full)
        if forecast_sarima and 'comparison' in forecast_sarima:
            sarima_df = forecast_sarima['comparison']
            sarima_pred = sarima_df.loc[actual_forecast.index, 'Predicted']
            mae = mean_absolute_error(actual_forecast, sarima_pred)
            rmse = np.sqrt(mean_squared_error(actual_forecast, sarima_pred))
            mape = np.mean(np.abs((actual_forecast - sarima_pred) / actual_forecast)) * 100
            full_metrics['SARIMA Full'] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

        # LSTM (Full)
        if lstm_forecast_df is not None:
            lstm_pred = lstm_forecast_df.loc[actual_forecast.index, 'Predicted']
            mae = mean_absolute_error(actual_forecast, lstm_pred)
            rmse = np.sqrt(mean_squared_error(actual_forecast, lstm_pred))
            mape = np.mean(np.abs((actual_forecast - lstm_pred) / actual_forecast)) * 100
            full_metrics['LSTM Full'] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
    else:
        st.info("ไม่มีข้อมูล Actual สำหรับช่วง Forecast 30 วัน (Full Data)")

    # --------------------------------
    # แสดง Metrics (Full Data)
    # --------------------------------
    if full_metrics:
        st.subheader("Metrics (Full Data Model)")
        cols_full = st.columns(4)
        order_full = ["Exp. Smoothing Full", "Moving Average Full", "SARIMA Full", "LSTM Full"]
        for i, mname in enumerate(order_full):
            with cols_full[i]:
                if mname in full_metrics:
                    mae_  = full_metrics[mname]['MAE']
                    rmse_ = full_metrics[mname]['RMSE']
                    mape_ = full_metrics[mname]['MAPE']
                    st.markdown(render_historical_metric_card(mname, mae_, rmse_, mape_), unsafe_allow_html=True)
                else:
                    st.markdown(render_historical_metric_card(mname, "-", "-", "-"), unsafe_allow_html=True)

    st.markdown("---")


if __name__ == "__main__":
    main()
