import pandas as pd
from pymavlink import mavutil

def extract_fusion_data_from_bin(file_path):
    """
    Mengekstrak dan mensinkronkan pesan ATT, MAG, dan IMU dari file log .BIN
    agar formatnya sama persis dengan Fusion_Data.csv.
    """
    try:
        mlog = mavutil.mavlink_connection(file_path)
    except Exception as e:
        raise ValueError(f"Gagal membaca file .BIN. Error: {e}")

    # State Tracker untuk menampung nilai terakhir dari sensor IMU dan MAG
    # yang frekuensinya berbeda dengan ATT.
    state = {
        'MagX': 0.0, 'MagY': 0.0, 'MagZ': 0.0,
        'abGyrX': 0.0, 'abGyrY': 0.0, 'abGyrZ': 0.0,
        'abAccX': 0.0, 'abAccY': 0.0, 'abAccZ': 0.0
    }
    
    data = []
    
    while True:
        msg = mlog.recv_match(type=['ATT', 'MAG', 'IMU'], blocking=False)
        if msg is None:
            break
            
        msg_dict = msg.to_dict()
        msg_type = msg.get_type()
        
        if msg_type == 'MAG':
            state['MagX'] = msg_dict.get('MagX', state['MagX'])
            state['MagY'] = msg_dict.get('MagY', state['MagY'])
            state['MagZ'] = msg_dict.get('MagZ', state['MagZ'])
            
        elif msg_type == 'IMU':
            state['abGyrX'] = msg_dict.get('GyrX', state['abGyrX'])
            state['abGyrY'] = msg_dict.get('GyrY', state['abGyrY'])
            state['abGyrZ'] = msg_dict.get('GyrZ', state['abGyrZ'])
            state['abAccX'] = msg_dict.get('AccX', state['abAccX'])
            state['abAccY'] = msg_dict.get('AccY', state['abAccY'])
            state['abAccZ'] = msg_dict.get('AccZ', state['abAccZ'])
            
        elif msg_type == 'ATT':
            # Jika ada pesan ATT, kita jadikan sebagai anchor sinkronisasi
            row = {
                'timestamp': msg_dict.get('TimeUS', 0),
                'DesRoll': msg_dict.get('DesRoll', 0.0),
                'Roll': msg_dict.get('Roll', 0.0),
                'DesPitch': msg_dict.get('DesPitch', 0.0),
                'Pitch': msg_dict.get('Pitch', 0.0),
                'DesYaw': msg_dict.get('DesYaw', 0.0),
                'Yaw': msg_dict.get('Yaw', 0.0),
                'ErrRP': msg_dict.get('ErrRP', 0.0),
                'ErrYaw': msg_dict.get('ErrYaw', 0.0),
                'MagX': state['MagX'],
                'MagY': state['MagY'],
                'MagZ': state['MagZ'],
                'abGyrX': state['abGyrX'],
                'abGyrY': state['abGyrY'],
                'abGyrZ': state['abGyrZ'],
                'abAccX': state['abAccX'],
                'abAccY': state['abAccY'],
                'abAccZ': state['abAccZ']
            }
            data.append(row)
            
    if not data:
        raise ValueError("Tidak ditemukan data ATT, MAG, atau IMU dalam file ini.")
        
    df = pd.DataFrame(data)
    
    # Tambahkan kolom waktu detik untuk kemudahan plot di visualisasi
    if 'timestamp' in df.columns and not df.empty:
        df['Time_Sec'] = (df['timestamp'] - df['timestamp'].min()) / 1e6
        
    # [PENTING UNTUK WINDOWS] Tutup file connection agar bisa dihapus oleh OS
    if hasattr(mlog, 'close'):
        mlog.close()
        
    return df
