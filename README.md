# AICUP - 太陽能發電量預測專案

## 📋 專案概述

本專案是參與台灣人工智慧盃 (AICUP) 競賽的太陽能發電量預測系統，使用深度學習技術預測太陽能板在不同氣象條件下的發電量。透過 LSTM (Long Short-Term Memory) 神經網路和線性回歸模型，結合氣象數據（風速、氣壓、溫度、濕度、日照強度）來預測未來 48 小時的太陽能發電量。

## 🎯 專案目標

- 開發準確的太陽能發電量預測模型
- 利用時間序列深度學習技術處理氣象數據
- 提供 48 小時的發電量預測
- 比較不同模型的預測效果

## 📊 資料特徵

本專案使用的資料包含以下氣象參數：

| 特徵 | 單位 | 描述 |
|------|------|------|
| WindSpeed | m/s | 風速 |
| Pressure | hpa | 氣壓 |
| Temperature | °C | 溫度 |
| Humidity | % | 濕度 |
| Sunlight | Lux | 日照強度 |
| Power | mW | 發電量（預測目標） |

## 🏗️ 專案結構

```
AICUP/
├── README.md                           # 專案說明文件
├── AICUP/
│   ├── aicup.py                       # 主要 LSTM 模型訓練腳本
│   └── taiwan_ai_cup_lstm.h5          # 訓練好的 LSTM 模型
├── final/
│   ├── final.py                       # 模型評估與比較腳本
│   ├── taiwan_ai_cup_lstm.h5          # LSTM 模型副本
│   ├── WeatherLSTM_input.h5           # 天氣預測 LSTM 模型
│   └── ExampleTrainData(AVG)/         # 平均訓練數據
├── LSTM test/
│   ├── Regression+LSTM.py             # 回歸+LSTM 組合模型
│   ├── WeatherLSTM_input.h5           # 天氣 LSTM 模型
│   ├── WeatherRegression_input        # 線性回歸模型
│   ├── output.csv                     # 預測結果輸出
│   └── ExampleTrainData(IncompleteAVG)/ # 不完整平均訓練數據
```

## 🤖 模型架構

### 1. LSTM 模型 (`aicup.py`)
- **輸入層**: 5 個氣象特徵，回溯 12 個時間步
- **LSTM 層**: 128 單元 (return_sequences=True) + 64 單元
- **Dropout 層**: 0.2 防止過擬合
- **輸出層**: Dense 層，輸出 48 個預測值

### 2. 混合模型 (`Regression+LSTM.py`)
- **LSTM 模型**: 用於天氣參數預測
- **線性回歸模型**: 用於發電量預測
- **組合預測**: 結合兩種方法提高準確性

## 🚀 使用方法

### 環境需求

```bash
pip install tensorflow
pip install scikit-learnㄊ
pip install pandas
pip install numpy
pip install joblib
```

### 訓練模型

1. **基礎 LSTM 模型訓練**:
```bash
cd AICUP
python aicup.py
```

2. **混合模型訓練**:
```bash
cd "LSTM test"
python "Regression+LSTM.py"
```

### 模型評估

```bash
cd final
python final.py
```

## 📈 模型參數

| 參數 | 值 | 說明 |
|------|-----|------|
| look_back_num | 12 | LSTM 回溯時間步數 |
| forecast_num | 48 | 預測未來時間步數 |
| epochs | 50-100 | 訓練週期 |
| batch_size | 128 | 批次大小 |
| optimizer | adam | 優化器 |
| loss | mse | 損失函數 |

## 📊 模型效能

專案包含兩個主要模型的比較：

1. **taiwan_ai_cup_lstm.h5**: 基礎 LSTM 模型
2. **WeatherLSTM_input.h5**: 進階天氣預測模型

使用 MSE (Mean Squared Error) 和準確度作為評估指標。

## 📁 資料格式

- **完整訓練資料**: `ExampleTrainData(AVG)/AvgDATA_XX.csv`
- **不完整訓練資料**: `ExampleTrainData(IncompleteAVG)/IncompleteAvgDATA_XX.csv`
- **輸出格式**: CSV 檔案包含預測的發電量數值

## 🔧 技術特點

- **時間序列預測**: 使用 LSTM 處理序列相關性
- **資料正規化**: MinMaxScaler 標準化輸入特徵
- **多步預測**: 一次預測 48 個時間步
- **模型比較**: 提供多種模型效能評估
- **可擴展性**: 支援多個地點的資料處理

## 📝 注意事項

- 確保資料檔案的路徑設定正確
- 模型訓練需要足夠的計算資源和時間
- 資料品質會直接影響預測準確性
- 建議使用 GPU 加速訓練過程
