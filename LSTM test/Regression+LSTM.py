import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import joblib
from datetime import datetime

#========================= 全局參數設置 =========================
LOOK_BACK_NUM = 12  # LSTM 模型的回溯步數
FORECAST_NUM = 48   # 預測步數
TRAIN_DATA_PATH = os.path.join(os.getcwd(), 'ExampleTrainData(AVG)', 'AvgDATA_17.csv')
TEST_DATA_PATH = os.path.join(os.getcwd(), 'ExampleTestData', 'upload.csv')

#========================= 資料載入與預處理 =========================
# 載入訓練資料
train_data = pd.read_csv(TRAIN_DATA_PATH, encoding='utf-8')

# 特徵與標籤選擇 (LSTM & 回歸分析)
features = ['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)']
target = ['Power(mW)']

# LSTM 資料正規化
scaler = MinMaxScaler().fit(train_data[features])
normalized_features = scaler.transform(train_data[features])

# 構建 LSTM 訓練集
X_train, y_train = [], []
for i in range(LOOK_BACK_NUM, len(normalized_features) - FORECAST_NUM + 1):  # 確保 y_train 的維度為 (batch_size, 48)
    X_train.append(normalized_features[i-LOOK_BACK_NUM:i])
    y_train.append(normalized_features[i:i+FORECAST_NUM, 0])  # 調整 y_train 的形狀，確保它有 48 步

X_train, y_train = np.array(X_train), np.array(y_train)

#========================= LSTM 模型構建與訓練 =========================
lstm_model = Sequential([
    LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(units=64),
    Dropout(0.2),
    Dense(units=FORECAST_NUM)  # 確保輸出是 48 步預測
])

lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_train, y_train, epochs=100, batch_size=128)

# 保存 LSTM 模型
lstm_model.save('WeatherLSTM_input.h5')
print("LSTM 模型保存完成")

#========================= 線性回歸模型構建與訓練 =========================
regression_X_train = scaler.transform(train_data[features])
regression_y_train = train_data[target].values

linear_model = LinearRegression()
linear_model.fit(regression_X_train, regression_y_train)

# 保存線性回歸模型
joblib.dump(linear_model, 'WeatherRegression_input')
print("線性回歸模型保存完成")

#========================= 測試資料處理與預測 =========================
# 載入測試資料
test_data = pd.read_csv(TEST_DATA_PATH, encoding='utf-8')
serial_numbers = test_data[['序號']].values.flatten()

lstm_model = load_model('WeatherLSTM_input.h5')
linear_model = joblib.load('WeatherRegression_input')

predictions, power_predictions = [], []
for index in range(1, 18):  # 從 1 到 17 迴圈
    # 根據 index 構建檔案名稱
    str_location_code = f"{index:02d}"  # 保證編號為兩位數 (01 到 17)
    file_name = f"IncompleteAvgDATA_{str_location_code}.csv"  # 構建檔案名稱
    data_path = os.path.join(os.getcwd(), 'ExampleTrainData(IncompleteAVG)', file_name)  # 生成完整路徑

    # 讀取對應天氣數據
    source_data = pd.read_csv(data_path, encoding='utf-8')
    reference_features = source_data[features]  # 使用 DataFrame 保持列名稱
    reference_features_scaled = scaler.transform(reference_features)  # 使用有列名稱的 DataFrame
    
    # 預測天氣參數與發電量
    inputs = []
    for day in range(LOOK_BACK_NUM, LOOK_BACK_NUM + FORECAST_NUM):
        test_input = reference_features_scaled[day - LOOK_BACK_NUM:day]
        test_input = np.reshape(test_input, (1, test_input.shape[0], test_input.shape[1]))  # 確保維度匹配
        prediction = lstm_model.predict(test_input)  # LSTM 預測 (1, forecast_num)
        predictions.append(prediction)  # 將預測結果添加到 predictions

        # 使用線性回歸模型來預測瓦數，這應該是單一瓦數預測
        power_prediction = linear_model.predict(reference_features_scaled[day - LOOK_BACK_NUM:day]).flatten()  # 單一預測值
        power_predictions.append(power_prediction)  # 添加單一瓦數預測到 power_predictions

# 確保 power_predictions 是一個包含單一列數據的列表
# 檢查 power_predictions 的形狀以確保它是 1D 數據
power_predictions = np.array(power_predictions).flatten()

# 創建結果 DataFrame
result_df = pd.DataFrame(power_predictions, columns=['Predicted Power (mW)'])
result_df.to_csv('output.csv', index=False)
print("預測結果保存完成")

