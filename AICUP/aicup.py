import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 設定檔案位置與合併
file_path = './'  # 資料檔案所在資料夾
file_prefix = 'L'
file_suffix = '_Train.csv'
num_files = 17

# 合併所有訓練資料
dataframes = []
for i in range(1, num_files + 1):
    filename = f"{file_path}{file_prefix}{i}{file_suffix}"
    df = pd.read_csv(filename)
    dataframes.append(df)

# 合併資料
data = pd.concat(dataframes, ignore_index=True)

# 保留需要的欄位
columns = ['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)', 'Power(mW)']
data = data[columns]

# 時間序列資料預處理
look_back_num = 12  # LSTM往前看的筆數
forecast_num = 48   # 預測筆數

# 正規化
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# 建立 LSTM 訓練資料
X_train, y_train = [], []
for i in range(look_back_num, len(data_scaled) - forecast_num):
    X_train.append(data_scaled[i - look_back_num:i, :-1])  # 特徵數據
    y_train.append(data_scaled[i:i + forecast_num, -1])   # 發電瓦數

X_train, y_train = np.array(X_train), np.array(y_train)

# LSTM 模型建構
model = Sequential()
model.add(LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=64))
model.add(Dropout(0.2))
model.add(Dense(units=forecast_num))  # 預測 48 筆

model.compile(optimizer='adam', loss='mean_squared_error')

# 模型訓練
model.fit(X_train, y_train, epochs=50, batch_size=128, verbose=1)

# 保存模型
model.save('taiwan_ai_cup_lstm.h5')
print("Model training and saving complete.")
