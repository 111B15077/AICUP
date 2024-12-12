import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# 設定檔案位置
data_path = './ExampleTrainData(AVG)'  # 資料夾位置
file_prefix = 'AvgDATA_'
file_suffix = '.csv'
num_files = 17

# 模型載入
model_path_1 = 'taiwan_ai_cup_lstm.h5'  # 第一個模型
model_path_2 = 'WeatherLSTM_input.h5'  # 第二個模型
model_1 = load_model(model_path_1)
model_2 = load_model(model_path_2)

# 設定特徵欄位與標籤
features = ['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)']
target = 'Power(mW)'

# 初始化結果儲存
mse_model_1 = []
mse_model_2 = []

for i in range(1, num_files + 1):
    filename = f"{file_prefix}{i:02d}{file_suffix}"
    filepath = os.path.join(data_path, filename)
    
    # 讀取測試資料
    test_data = pd.read_csv(filepath)
    
    # 確認資料是否完整
    if not all(col in test_data.columns for col in features + [target]):
        print(f"檔案 {filename} 缺少必要欄位，跳過處理。")
        continue

    # 特徵正規化
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(test_data[features + [target]])
    test_features = scaled_data[:, :-1]
    test_target = scaled_data[:, -1]
    
    # 構建測試資料集
    look_back_num = 12  # 根據模型設定
    forecast_num = 48   # 預測步數
    X_test, y_test = [], []
    for j in range(look_back_num, len(test_features) - forecast_num + 1):
        X_test.append(test_features[j-look_back_num:j])  # 過去 look_back_num 筆資料作為輸入
        y_test.append(test_target[j:j + forecast_num])   # 預測接下來的 forecast_num 筆

    X_test, y_test = np.array(X_test), np.array(y_test)

    # 確認形狀
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # 模型預測
    y_pred_1 = model_1.predict(X_test)  # 預測值形狀應為 (測試樣本數, forecast_num)
    y_pred_2 = model_2.predict(X_test)

    # 打印預測結果的形狀，檢查每個預測結果的步數
    print(f"y_pred_1 shape: {y_pred_1.shape}")
    print(f"y_pred_2 shape: {y_pred_2.shape}")

    # 計算每次預測的 MSE 並計算準確度
    for k in range(len(y_test)):
        mse_1 = mean_squared_error(y_test[k], y_pred_1[k])  # 檢查輸出是否一致
        mse_2 = mean_squared_error(y_test[k], y_pred_2[k])
        
        # 計算準確度
        accuracy_1 = (1 - mse_1) * 100
        accuracy_2 = (1 - mse_2) * 100
        
        mse_model_1.append(mse_1)
        mse_model_2.append(mse_2)

        print(f"模型1 (MSE: {mse_1:.4f}, Accuracy: {accuracy_1:.2f}%)")
        print(f"模型2 (MSE: {mse_2:.4f}, Accuracy: {accuracy_2:.2f}%)")

# 計算平均MSE和準確度
avg_mse_1 = np.mean(mse_model_1)
avg_mse_2 = np.mean(mse_model_2)

# 計算總體準確度
avg_accuracy_1 = (1 - avg_mse_1) * 100
avg_accuracy_2 = (1 - avg_mse_2) * 100

print(f"模型1 (taiwan_ai_cup_lstm.h5) 平均MSE: {avg_mse_1:.4f}, 平均準確度: {avg_accuracy_1:.2f}%")
print(f"模型2 (WeatherLSTM_input.h5) 平均MSE: {avg_mse_2:.4f}, 平均準確度: {avg_accuracy_2:.2f}%")
