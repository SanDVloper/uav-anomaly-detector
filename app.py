import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import tempfile
import os
import pickle
from tensorflow.keras.models import load_model

# Import parser sinkronisasi Multi-Sensor (IMU, ATT, MAG) dari .BIN
import parser_utils

# ==========================================
# FUNGSI PREDIKSI
# ==========================================
@st.cache_resource
def load_ml_components():
    """
    Memuat model Jaringan Saraf Tiruan (model.h5) dan Scaler (scaler.pkl).
    """
    try:
        model = load_model('model.h5')
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Gagal memuat model.h5 atau scaler.pkl. Pastikan Anda telah menjalankan train_model.ipynb. Error detail: {e}")
        return None, None

def predict_anomalies(df, model, scaler):
    """
    Melakukan prediksi anomali pada data sensor fusion.
    """
    # Fitur wajib Multi-Sensor (sesuai Fusion_Data.csv)
    feature_cols = [
        'DesRoll', 'Roll', 'DesPitch', 'Pitch', 'DesYaw', 'Yaw', 'ErrRP', 'ErrYaw',
        'MagX', 'MagY', 'MagZ',
        'abGyrX', 'abGyrY', 'abGyrZ', 'abAccX', 'abAccY', 'abAccZ'
    ]
    
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Fitur dataset tidak lengkap. Kolom yang hilang: {missing_cols}")
        
    X = df[feature_cols]
    
    # Standarisasi nilai sensor
    X_scaled = scaler.transform(X)
    
    # Prediksi ANN
    predictions = model.predict(X_scaled)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # 0 = Normal, 1-4 = Anomali
    df['Prediction_Class'] = predicted_classes
    df['Is_Anomaly'] = df['Prediction_Class'].apply(lambda x: 1 if x > 0 else 0)
    
    return df

# ==========================================
# UI STREAMLIT (DASHBOARD)
# ==========================================
def main():
    st.set_page_config(page_title="UAV Multi-Sensor Anomaly", layout="wide", page_icon="🚁")
    
    st.title("Sistem Otomatis Deteksi Anomali UAV (Sensor Fusion)")
    st.markdown("""
    Aplikasi ini menganalisis file log **Pixhawk (.BIN)** dan mengekstrak pesan Orientasi, IMU, serta Magnetometer.
    AI berbasis Multi-Layer Perceptron akan mensinkronkan lalu memonitor pergerakan mekanis drone Anda (Sensor Fusion) untuk mendeteksi kegagalan.
    """)
    
    model, scaler = load_ml_components()
    if model is None or scaler is None:
        st.warning("Menunggu model dan scaler... Silakan jalankan `train_model.ipynb` terlebih dahulu.")
        st.stop()
        
    st.sidebar.header("Unggah File Log (.BIN)")
    uploaded_file = st.sidebar.file_uploader("", type=['bin', 'BIN'])
    
    if uploaded_file is not None:
        with st.spinner('Mem-parsing file .BIN dan Mensinkronisasi IMU/ATT/MAG (Harap tunggu)...'):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".BIN") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                    
                # Ekstrak data Multi-Sensor!
                df_fused = parser_utils.extract_fusion_data_from_bin(tmp_file_path)
                
                os.remove(tmp_file_path)
            except Exception as e:
                st.error(f"Gagal memproses file log: {e}")
                return
                
        with st.spinner('Mendeteksi anomali menggunakan Neural Network...'):
            try:
                df_results = predict_anomalies(df_fused, model, scaler)
            except Exception as e:
                st.error(f"Gagal memprediksi data: {e}")
                return
                
        # Dasbor Ringkasan
        total_duration_sec = df_results['Time_Sec'].max() if 'Time_Sec' in df_results.columns else 0
        total_rows = len(df_results)
        anomaly_count = df_results['Is_Anomaly'].sum()
        anomaly_percentage = (anomaly_count / total_rows) * 100 if total_rows > 0 else 0
        
        st.markdown("Ringkasan Sensor Fusion & Prediksi")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Durasi Sinkronisasi", f"{total_duration_sec:.2f} Detik", f"{(total_duration_sec/60):.2f} Menit")
        with col2:
            st.metric("Total Sampel (Baris)", f"{total_rows:,} Baris")
        with col3:
            d_color = "normal" if anomaly_percentage == 0 else "inverse"
            delta_text = "Status: Mekanis/Kontrol Normal" if anomaly_percentage == 0 else f"Peringatan: {anomaly_count} titik bahaya!"
            st.metric("Persentase Anomali", f"{anomaly_percentage:.2f} %", delta_text, delta_color=d_color)
            
        st.divider()
        
        # Visualisasi Time-Series IMU / ATT
        st.markdown("Visualisasi Time-Series: Deteksi Anomali Orientasi")
        
        y_axis_option = st.selectbox(
            "Pilih Parameter Sensor untuk Sumbu Y:", 
            options=['Roll (Kemiringan Kiri/Kanan)', 'Pitch (Kemiringan Depan/Belakang)', 'Yaw (Arah Hadap)', 'abGyrZ (Rotasi Z)']
        )
        
        y_column = y_axis_option.split(" ")[0]
        
        fig = go.Figure()
        
        df_normal = df_results[df_results['Is_Anomaly'] == 0]
        df_anomaly = df_results[df_results['Is_Anomaly'] == 1]
        
        fig.add_trace(go.Scatter(
            x=df_normal['Time_Sec'], y=df_normal[y_column],
            mode='markers', marker=dict(color='#2ca02c', size=4, opacity=0.7),
            name='Normal'
        ))
        
        if not df_anomaly.empty:
            fig.add_trace(go.Scatter(
                x=df_anomaly['Time_Sec'], y=df_anomaly[y_column],
                mode='markers', marker=dict(color='#d62728', size=10, symbol='x'),
                name='Anomali',
                text=df_anomaly['Prediction_Class'].apply(lambda x: f"Jenis Anomali: Tipe {x}"),
                hoverinfo='text+x+y'
            ))
            
        fig.update_layout(
            title=f"Pergerakan {y_axis_option} vs Waktu",
            xaxis_title="Waktu Sinkronisasi (Detik)",
            yaxis_title=f"Nilai {y_column}",
            hovermode="closest",
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("Tampilkan Data Tabel Sensor Fusion (Raw)"):
            st.dataframe(df_results)

if __name__ == "__main__":
    main()
