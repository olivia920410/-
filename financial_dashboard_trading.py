# -*- coding: utf-8 -*-
"""
金融資料視覺化看板

@author: 
"""

# 載入必要模組
import os
#import haohaninfo
#from order_Lo8 import Record
import numpy as np
#from talib.abstract import SMA,EMA, WMA, RSI, BBANDS, MACD
#import sys
import requests,datetime,os,time
import indicator_f_Lo2_short,datetime, indicator_forKBar_short
import pandas as pd
import streamlit as st 
import streamlit.components.v1 as stc 
from order_streamlit import Record
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
import datetime
#%%
####### (1) 開始設定 #######
###### 設定網頁標題介面 
html_temp = """
		<div style="background-color:#3872fb;padding:10px;border-radius:10px">   
		<h1 style="color:white;text-align:center;">金融看板與程式交易平台 </h1>
		<h2 style="color:white;text-align:center;">Financial Dashboard and Program Trading </h2>
		</div>
		"""
stc.html(html_temp)


###### 讀取資料
@st.cache_data(ttl=3600, show_spinner="正在加載資料...")  ## Add the caching decorator
def load_data(path):
    df = pd.read_pickle(path)
    return df
# ##### 讀取 excel 檔
# df_original = pd.read_excel("kbars_2330_2022-01-01-2022-11-18.xlsx")


###### 選擇金融商品
st.subheader("選擇金融商品: ")
# choices = ['台積電: 2022.1.1 至 2024.4.9', '大台指2024.12到期: 2024.1 至 2024.4.9']
54
choices = ['中鋼期貨: 2023.4.17 至 2025.4.17', '聯電期貨: 2023.4.17 至 2025.4.17', '台積電期貨: 2023.4.17 至 2025.4.17', '富邦金期貨: 2023.4.17 至 2025.4.17', '台新金期貨: 2023.4.17 至 2025.4.17'
           ,'統一期貨: 2023.4.17 至 2025.4.17','金融期貨: 2023.4.17 至 2025.4.17','小型臺指期: 2023.4.17 至 2025.4.17','臺股期貨: 2023.4.17 至 2025.4.17','元大台灣50: 2023.4.17 至 2025.4.17','元大台灣50正2: 2023.4.17 至 2025.4.17'
           ,'台積電: 2023.4.17 至 2025.4.17','華碩: 2023.4.17 至 2025.4.17']
choice = st.selectbox('選擇金融商品', choices, index=0)

##### 讀取Pickle文件並轉換 'time' 欄位
# 將轉換邏輯統一處理，避免重複
data_paths = {
    '中鋼期貨: 2023.4.17 至 2025.4.17': ('kbars_1d_CBF_2023-04-17_To_2025-04-17.pkl', '中鋼期貨CBF'),
    '聯電期貨: 2023.4.17 至 2025.4.17': ('kbars_1d_CCF_2023-04-17_To_2025-04-17', '聯電期貨CCF'), # 注意這個沒有 .pkl
    '台積電期貨: 2023.4.17 至 2025.4.17': ('kbars_1d_CDF_2023-04-17_To_2025-04-17.pkl', '台積電期貨CDF'),
    '富邦金期貨: 2023.4.17 至 2025.4.17': ('kbars_1d_CEF_2023-04-17_To_2025-04-17.pkl', '富邦金期貨CEF'),
    '台新金期貨: 2023.4.17 至 2025.4.17': ('kbars_1d_CMF_2023-04-17_To_2025-04-17.pkl', '台新金期貨CMF'),
    '統一期貨: 2023.4.17 至 2025.4.17': ('kbars_1d_CQF_2023-04-17_To_2025-04-17.pkl', '統一期貨CQF'),
    '金融期貨: 2023.4.17 至 2025.4.17': ('kbars_1d_FXF_2023-04-17_To_2025-04-17.pkl', '金融期貨FXF'),
    '小型臺指期: 2023.4.17 至 2025.4.17': ('kbars_1d_MXF_2023-04-17_To_2025-04-17.pkl', '小型臺指期MXF'),
    '臺股期貨: 2023.4.17 至 2025.4.17': ('kbars_1d_TXF_2023-04-17_To_2025-04-17.pkl', '臺股期貨TXF'),
    '元大台灣50: 2023.4.17 至 2025.4.17': ('kbars_1d_0050_2023-04-17_To_2025-04-17.pkl', '元大台灣50 0050'),
    '元大台灣50正2: 2023.4.17 至 2025.4.17': ('kbars_1d_00631L_2023-04-17_To_2025-04-17.pkl', '元大台灣50正2 00631L'),
    '台積電: 2023.4.17 至 2025.4.17': ('kbars_1d_2330_2023-04-17_To_2025-04-17.pkl', '台積電2330'),
    '華碩: 2023.4.17 至 2025.4.17': ('kbars_1d_2357_2023-04-17_To_2025-04-17.pkl', '華碩2357')
}
st.write("--- Debug Point A: Before loading df_original ---")
st.write(f"Current choice: {choice}")
st.write(f"Is choice in data_paths? {choice in data_paths}")

if choice in data_paths:
    path, name = data_paths[choice]
    df_original = load_data(path)
    product_name = name
    # --- 關鍵修正點：在載入數據後，立即轉換 'time' 欄位 ---
    df_original['time'] = pd.to_datetime(df_original['time'])
    # 如果你的 Pickle 文件中有 'Unnamed: 0' 這樣的無用欄位，可以在這裡刪除
    if 'Unnamed: 0' in df_original.columns:
        df_original = df_original.drop('Unnamed: 0', axis=1)
    st.write("--- Debug Point B: df_original loaded successfully ---")
    st.write(f"df_original['time'] dtype: {df_original['time'].dtype}")
    st.write(f"product_name: {product_name}")
else:
    st.error("請選擇一個有效的金融商品。")
    st.stop()



###### 選擇資料區間
st.subheader("選擇資料時間區間")

# 簡化所有 choice 條件下的重複程式碼
# 無論選擇哪個 choice，start_date_str 和 end_date_str 的設定邏輯都是一樣的
start_date_str = st.text_input('輸入開始日期(日期格式: 2023-04-17), 區間:2023-04-17 至 2025-04-17', '2023-04-17')
end_date_str = st.text_input('輸入結束日期 (日期格式: 2025-04-17), 區間:2023-04-17 至 2025-04-17', '2025-04-17')
st.write("--- Debug Point C: After date string input ---")
st.write(f"start_date_str: {start_date_str}")
st.write(f"end_date_str: {end_date_str}")
# 2. 嘗試將字串轉換為 datetime 物件
try:
    # 這裡使用 %Y-%m-%d，因為你的預設值和提示都是這種格式
    start_date_dt = datetime.datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date_dt = datetime.datetime.strptime(end_date_str, '%Y-%m-%d')
    start_date = start_date_dt
    end_date = end_date_dt
    # --- 定義 Date 變數 ---
    Date = start_date # <--- 這是您要確保被執行的行

    st.write("--- Debug Point D: Dates parsed and Date variable defined ---")
    st.write(f"start_date type: {type(start_date)}, value: {start_date}")
    st.write(f"end_date type: {type(end_date)}, value: {end_date}")
    st.write(f"Date type: {type(Date)}, value: {Date}")
except ValueError as e:
    st.error(f"日期格式不符！請輸入正確格式的日期 (例如 2023-04-17)。錯誤詳情: {e}")
    st.stop() # <-- 偵測點：如果程式在此停止，表示日期格式有問題。

# --- 診斷程式碼 (確認型別) ---
st.write(f"df_original['time'] 的資料型別: {df_original['time'].dtype}")
st.write(f"start_date 的資料型別: {type(start_date)}, 值: {start_date}")
st.write(f"end_date 的資料型別: {type(end_date)}, 值: {end_date}")
# --- 診斷程式碼結束 ---


# 在 financial_dashboard_trading.py 的 line 125 之前加入以下程式碼：

st.write(f"df_original['time'] 的資料型別: {df_original['time'].dtype}")
st.write(f"start_date 的資料型別: {type(start_date)}, 值: {start_date}")
st.write(f"end_date 的資料型別: {type(end_date)}, 值: {end_date}")

# 然後是導致錯誤的這行：
# df = df_original[(df_original['time'] >= start_date) & (df_original['time'] <= end_date)]
# --- 關鍵修正點 ---
# 使用 datetime 物件來篩選 DataFrame，而不是字串
df = df_original[(df_original['time'] >= start_date) & (df_original['time'] <= end_date)]
# --- 修正結束 ---
# 使用 datetime 物件來篩選 DataFrame
df = df_original[(df_original['time'] >= start_date) & (df_original['time'] <= end_date)]

st.write("--- Debug Point E: DataFrame filtered ---")
st.write(f"Filtered df shape: {df.shape}")

#%%
####### (2) 轉化為字典 #######
@st.cache_data(ttl=3600, show_spinner="正在加載資料...") ## Add the caching decorator
def To_Dictionary_1(df, product_name):
    KBar_dic = df.to_dict()

    # 確保 'time' 欄位中的每個元素都是 datetime.datetime 物件
    # 如果 df_original['time'] 已經是 datetime64[ns]，to_dict() 會將其轉換為 Timestamp 物件。
    # KBar.AddPrice 和 KBar.__init__ 最好處理原生 datetime.datetime 物件。
    KBar_time_list = [t.to_pydatetime() if isinstance(t, pd.Timestamp) else t for t in KBar_dic['time'].values()]
    KBar_dic['time'] = np.array(KBar_time_list)

    # 其餘的保持不變，將列表轉換為 numpy 數組
    KBar_open_list = list(KBar_dic['open'].values())
    KBar_dic['open'] = np.array(KBar_open_list)
    
    KBar_dic['product'] = np.repeat(product_name, KBar_dic['open'].size)
    
    KBar_low_list = list(KBar_dic['low'].values())
    KBar_dic['low'] = np.array(KBar_low_list)
    
    KBar_high_list = list(KBar_dic['high'].values())
    KBar_dic['high'] = np.array(KBar_high_list)
    
    KBar_close_list = list(KBar_dic['close'].values())
    KBar_dic['close'] = np.array(KBar_close_list)
    
    KBar_volume_list = list(KBar_dic['volume'].values())
    KBar_dic['volume'] = np.array(KBar_volume_list)
    
    KBar_amount_list = list(KBar_dic['amount'].values())
    KBar_dic['amount'] = np.array(KBar_amount_list)
    
    return KBar_dic

# ... (To_Dictionary_1 函數定義和 KBar_dic 呼叫) ...
KBar_dic = To_Dictionary_1(df, product_name)
st.write("--- Debug Point F: KBar_dic created ---")
st.write(f"KBar_dic['time'] size: {KBar_dic['time'].size}")


#%%
####### (3) 改變 KBar 時間長度 & 形成 KBar 字典 (新週期的) & Dataframe #######
###### 定義函數: 進行 K 棒更新  &  形成 KBar 字典 (新週期的): 設定cycle_duration可以改成你想要的 KBar 週期
@st.cache_data(ttl=3600, show_spinner="正在加載資料...") ## Add the caching decorator
def Change_Cycle(Date,cycle_duration,KBar_dic):
    ###### 進行 K 棒更新
    # 這裡的 Date (也就是你 financial_dashboard_trading.py 中的 start_date_dt)
    # 已經是 datetime.datetime 物件，這是正確的。
    # 確保你的 indicator_forKBar_short.py 中的 KBar 類別的 __init__ 方法能處理 datetime 物件。
    KBar = indicator_forKBar_short.KBar(Date,cycle_duration)
    
    for i in range(KBar_dic['time'].size):
        # time 變數現在應該是 datetime.datetime 物件，這是 KBar.AddPrice 所期望的
        time = KBar_dic['time'][i]
        open_price= KBar_dic['open'][i]
        close_price= KBar_dic['close'][i]
        low_price= KBar_dic['low'][i]
        high_price= KBar_dic['high'][i]
        qty = KBar_dic['volume'][i]
        
        KBar.AddPrice(time, open_price, close_price, low_price, high_price, qty)
    
    ###### 形成 KBar 字典 (新週期的):
    KBar_dic_new_cycle = {} # 使用一個新的變數名，避免混淆
    KBar_dic_new_cycle['time'] = KBar.TAKBar['time']    
    KBar_dic_new_cycle['product'] = np.repeat('tsmc', KBar_dic_new_cycle['time'].size) # 假設都是 'tsmc'
    KBar_dic_new_cycle['open'] = KBar.TAKBar['open']
    KBar_dic_new_cycle['high'] = KBar.TAKBar['high']
    KBar_dic_new_cycle['low'] = KBar.TAKBar['low']
    KBar_dic_new_cycle['close'] = KBar.TAKBar['close']
    KBar_dic_new_cycle['volume'] = KBar.TAKBar['volume']
    
    return KBar_dic_new_cycle # 返回新的字典

# ---

st.subheader("設定技術指標視覺化圖形之相關參數:")

# ... (KBar 週期設定的 expander 區塊) ...
# 請確保 cycle_duration 在這裡被定義
# 簡化後的 cycle_duration 賦值：
cycle_duration = 0 # 賦予一個初始值以防萬一
with st.expander("設定K棒相關參數:"):
    choices_unit = ['以分鐘為單位','以日為單位','以週為單位','以月為單位']
    choice_unit = st.selectbox('選擇計算K棒時間長度之單位', choices_unit, index=1)
    if choice_unit == '以分鐘為單位':
        cycle_duration = st.number_input('輸入一根 K 棒的時間長度(單位:分鐘, 一日=1440分鐘)', value=1, key="KBar_duration_分")
    elif choice_unit == '以日為單位': # 使用 elif
        cycle_duration = st.number_input('輸入一根 K 棒的時間長度(單位:日)', value=1, key="KBar_duration_日") * 1440
    elif choice_unit == '以週為單位': # 使用 elif
        cycle_duration = st.number_input('輸入一根 K 棒的時間長度(單位:週)', value=1, key="KBar_duration_週") * 7 * 1440
    elif choice_unit == '以月為單位': # 使用 elif
        cycle_duration = st.number_input('輸入一根 K 棒的時間長度(單位:月, 一月=30天)', value=1, key="KBar_duration_月") * 30 * 1440
    cycle_duration = float(cycle_duration) # 確保是浮點數

st.write("--- Debug Point G: Before Change_Cycle call ---")
st.write(f"Date is defined: {'Date' in locals()}") # 確認 Date 是否在這個作用域內
st.write(f"cycle_duration: {cycle_duration}")
st.write(f"KBar_dic is defined: {'KBar_dic' in locals()}")


###### 進行 K 棒更新  & 形成 KBar 字典 (新週期的)
# 期待 'Date', 'cycle_duration', 'KBar_dic' 都已在此定義
KBar_dic_new = Change_Cycle(Date,cycle_duration,KBar_dic) # Line 228 (您的錯誤行)

st.write("--- Debug Point H: Change_Cycle completed ---")
###### 將K線 Dictionary 轉換成 Dataframe
KBar_df = pd.DataFrame(KBar_dic_new) # 使用新的 KBar 字典來創建 DataFrame



#%%
####### (4) 計算各種技術指標 #######

#%%
######  (i) 移動平均線策略 
@st.cache_data(ttl=3600, show_spinner="正在加載資料...")  ## Add the caching decorator
def Calculate_MA(df, period=10):
    ##### 計算長短移動平均線
    ma = df['close'].rolling(window=period).mean()
    return ma
  
#####  設定長短移動平均線的 K棒 長度:
with st.expander("設定長短移動平均線的 K棒 長度:"):
    # st.subheader("設定計算長移動平均線(MA)的 K棒週期數目(整數, 例如 10)")
    LongMAPeriod=st.slider('設定計算長移動平均線(MA)的 K棒週期數目(整數, 例如 10)', 0, 100, 10, key='visualization_MA_long')
    # st.subheader("設定計算短移動平均線(MA)的 K棒週期數目(整數, 例如 2)")
    ShortMAPeriod=st.slider('設定計算短移動平均線(MA)的 K棒週期數目(整數, 例如 2)', 0, 100, 2, key='visualization_MA_short')

##### 計算長短移動平均線
KBar_df['MA_long'] = Calculate_MA(KBar_df, period=LongMAPeriod)
KBar_df['MA_short'] = Calculate_MA(KBar_df, period=ShortMAPeriod)

##### 尋找最後 NAN值的位置
last_nan_index_MA = KBar_df['MA_long'][::-1].index[KBar_df['MA_long'][::-1].apply(pd.isna)][0]


#%%
######  (ii) RSI 策略 
##### 假设 df 是一个包含价格数据的Pandas DataFrame，其中 'close' 是KBar週期收盤價
@st.cache_data(ttl=3600, show_spinner="正在加載資料...")  ## Add the caching decorator
def Calculate_RSI(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
  
##### 順勢策略
#### 設定長短 RSI 的 K棒 長度:
with st.expander("設定長短 RSI 的 K棒 長度:"):
    # st.subheader("設定計算長RSI的 K棒週期數目(整數, 例如 10)")
    LongRSIPeriod=st.slider('設定計算長RSI的 K棒週期數目(整數, 例如 10)', 0, 1000, 10, key='visualization_RSI_long')
    # st.subheader("設定計算短RSI的 K棒週期數目(整數, 例如 2)")
    ShortRSIPeriod=st.slider('設定計算短RSI的 K棒週期數目(整數, 例如 2)', 0, 1000, 2, key='visualization_RSI_short')

#### 計算 RSI指標長短線, 以及定義中線
KBar_df['RSI_long'] = Calculate_RSI(KBar_df, LongRSIPeriod)
KBar_df['RSI_short'] = Calculate_RSI(KBar_df, ShortRSIPeriod)
KBar_df['RSI_Middle'] = np.array([50] * len(KBar_df))

#### 尋找最後 NAN值的位置
last_nan_index_RSI = KBar_df['RSI_long'][::-1].index[KBar_df['RSI_long'][::-1].apply(pd.isna)][0]


# ##### 逆勢策略
# #### 建立部位管理物件
# OrderRecord=Record() 
# #### 計算 RSI指標, 天花板與地板
# RSIPeriod=5
# Ceil=80
# Floor=20
# MoveStopLoss=30
# KBar_dic['RSI']=RSI(KBar_dic,timeperiod=RSIPeriod)
# KBar_dic['Ceil']=np.array([Ceil]*len(KBar_dic['time']))
# KBar_dic['Floor']=np.array([Floor]*len(KBar_dic['time']))

# #### 將K線 Dictionary 轉換成 Dataframe
# KBar_RSI_df=pd.DataFrame(KBar_dic)


#%%
######  (iii) Bollinger Band (布林通道) 策略 
##### 假设df是包含价格数据的Pandas DataFrame，'close'列是每日收盘价格
@st.cache_data(ttl=3600, show_spinner="正在加載資料...")  ## Add the caching decorator
def Calculate_Bollinger_Bands(df, period=20, num_std_dev=2):
    df['SMA'] = df['close'].rolling(window=period).mean()
    df['Standard_Deviation'] = df['close'].rolling(window=period).std()
    df['Upper_Band'] = df['SMA'] + (df['Standard_Deviation'] * num_std_dev)
    df['Lower_Band'] = df['SMA'] - (df['Standard_Deviation'] * num_std_dev)
    return df


#####  設定布林通道(Bollinger Band)相關參數:
with st.expander("設定布林通道(Bollinger Band)相關參數:"):
    # st.subheader("設定計算布林通道(Bollinger Band)上中下三通道之K棒週期數目(整數, 例如 20)")
    period = st.slider('設定計算布林通道(Bollinger Band)上中下三通道之K棒週期數目(整數, 例如 20)', 0, 100, 20, key='BB_period')
    # st.subheader("設定計算布林通道(Bollinger Band)上中(或下中)通道之帶寬(例如 2 代表上中通道寬度為2倍的標準差)")
    num_std_dev = st.slider('設定計算布林通道(Bollinger Band)上中(或下中)通道之帶寬(例如 2 代表上中通道寬度為2倍的標準差)', 0, 100, 2, key='BB_heigh')

##### 計算布林通道上中下通道:
KBar_df = Calculate_Bollinger_Bands(KBar_df, period, num_std_dev)

##### 尋找最後 NAN值的位置
last_nan_index_BB = KBar_df['SMA'][::-1].index[KBar_df['SMA'][::-1].apply(pd.isna)][0]


#%%
######  (iv) MACD(異同移動平均線) 策略 
# 假设df是包含价格数据的Pandas DataFrame，'price'列是每日收盘价格
@st.cache_data(ttl=3600, show_spinner="正在加載資料...")  ## Add the caching decorator
def Calculate_MACD(df, fast_period=12, slow_period=26, signal_period=9):
    df['EMA_Fast'] = df['close'].ewm(span=fast_period, adjust=False).mean()
    df['EMA_Slow'] = df['close'].ewm(span=slow_period, adjust=False).mean()
    df['MACD'] = df['EMA_Fast'] - df['EMA_Slow']  ## DIF
    df['Signal_Line'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()   ## DEA或信號線
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']  ## MACD = DIF-DEA
    return df

#####  設定MACD三種週期的K棒長度:
with st.expander("設定MACD三種週期的K棒長度:"):
    # st.subheader("設定計算 MACD的快速線週期(例如 12根日K)")
    fast_period = st.slider('設定計算 MACD快速線的K棒週期數目(例如 12根日K)', 0, 100, 12, key='visualization_MACD_quick')
    # st.subheader("設定計算 MACD的慢速線週期(例如 26根日K)")
    slow_period = st.slider('設定計算 MACD慢速線的K棒週期數目(例如 26根日K)', 0, 100, 26, key='visualization_MACD_slow')
    # st.subheader("設定計算 MACD的訊號線週期(例如 9根日K)")
    signal_period = st.slider('設定計算 MACD訊號線的K棒週期數目(例如 9根日K)', 0, 100, 9, key='visualization_MACD_signal')

##### 計算MACD:
KBar_df = Calculate_MACD(KBar_df, fast_period, slow_period, signal_period)

##### 尋找最後 NAN值的位置
# last_nan_index_MACD = KBar_df['MACD'][::-1].index[KBar_df['MACD'][::-1].apply(pd.isna)][0]
#### 試著找出最後一個 NaN 值的索引，但在這之前要檢查是否有 NaN 值
nan_indexes_MACD = KBar_df['MACD'][::-1].index[KBar_df['MACD'][::-1].apply(pd.isna)]
if len(nan_indexes_MACD) > 0:
    last_nan_index_MACD = nan_indexes_MACD[0]
else:
    last_nan_index_MACD = 0




# ####### (5) 將 Dataframe 欄位名稱轉換(第一個字母大寫)  ####### 
# KBar_df_original = KBar_df
# KBar_df.columns = [ i[0].upper()+i[1:] for i in KBar_df.columns ]


#%%
####### (5) 畫圖 #######
st.subheader("技術指標視覺化圖形")
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
#from plotly.offline import plot
# import plotly.offline as pyoff


###### K線圖, 移動平均線MA
with st.expander("K線圖, 移動平均線"):
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    
    #### include candlestick with rangeselector
    fig1.add_trace(go.Candlestick(x=KBar_df['time'],
                    open=KBar_df['open'], high=KBar_df['high'],
                    low=KBar_df['low'], close=KBar_df['close'], name='K線'),
                   secondary_y=True)   ## secondary_y=True 表示此圖形的y軸scale是在右邊而不是在左邊
    
    #### include a go.Bar trace for volumes
    fig1.add_trace(go.Bar(x=KBar_df['time'], y=KBar_df['volume'], name='成交量', marker=dict(color='black')),secondary_y=False)  ## secondary_y=False 表示此圖形的y軸scale是在左邊而不是在右邊
    fig1.add_trace(go.Scatter(x=KBar_df['time'][last_nan_index_MA+1:], y=KBar_df['MA_long'][last_nan_index_MA+1:], mode='lines',line=dict(color='orange', width=2), name=f'{LongMAPeriod}-根 K棒 移動平均線'), 
                  secondary_y=True)
    fig1.add_trace(go.Scatter(x=KBar_df['time'][last_nan_index_MA+1:], y=KBar_df['MA_short'][last_nan_index_MA+1:], mode='lines',line=dict(color='pink', width=2), name=f'{ShortMAPeriod}-根 K棒 移動平均線'), 
                  secondary_y=True)
    
    fig1.layout.yaxis2.showgrid=True
    st.plotly_chart(fig1, use_container_width=True)


###### K線圖, RSI
with st.expander("長短 RSI"):
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    #### include candlestick with rangeselector
    # fig2.add_trace(go.Candlestick(x=KBar_df['Time'],
    #                 open=KBar_df['Open'], high=KBar_df['High'],
    #                 low=KBar_df['Low'], close=KBar_df['Close'], name='K線'),
    #                secondary_y=True)   ## secondary_y=True 表示此圖形的y軸scale是在右邊而不是在左邊
    
    fig2.add_trace(go.Scatter(x=KBar_df['time'][last_nan_index_RSI+1:], y=KBar_df['RSI_long'][last_nan_index_RSI+1:], mode='lines',line=dict(color='red', width=2), name=f'{LongRSIPeriod}-根 K棒 移動 RSI'), 
                  secondary_y=False)
    fig2.add_trace(go.Scatter(x=KBar_df['time'][last_nan_index_RSI+1:], y=KBar_df['RSI_short'][last_nan_index_RSI+1:], mode='lines',line=dict(color='blue', width=2), name=f'{ShortRSIPeriod}-根 K棒 移動 RSI'), 
                  secondary_y=False)
    
    fig2.layout.yaxis2.showgrid=True
    st.plotly_chart(fig2, use_container_width=True)
    

###### K線圖, Bollinger Band    
with st.expander("K線圖,布林通道"):
    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
    fig3.add_trace(go.Candlestick(x=KBar_df['time'],
                    open=KBar_df['open'], high=KBar_df['high'],
                    low=KBar_df['low'], close=KBar_df['close'], name='K線'),
                    secondary_y=True)    
    fig3.add_trace(go.Scatter(x=KBar_df['time'][last_nan_index_BB+1:], y=KBar_df['SMA'][last_nan_index_BB+1:], mode='lines',line=dict(color='black', width=2), name='布林通道中軌道'), 
                  secondary_y=False)
    fig3.add_trace(go.Scatter(x=KBar_df['time'][last_nan_index_BB+1:], y=KBar_df['Upper_Band'][last_nan_index_BB+1:], mode='lines',line=dict(color='red', width=2), name='布林通道上軌道'), 
                  secondary_y=False)
    fig3.add_trace(go.Scatter(x=KBar_df['time'][last_nan_index_BB+1:], y=KBar_df['Lower_Band'][last_nan_index_BB+1:], mode='lines',line=dict(color='blue', width=2), name='布林通道下軌道'), 
                  secondary_y=False)
    
    fig3.layout.yaxis2.showgrid=True

    st.plotly_chart(fig3, use_container_width=True)



###### MACD
with st.expander("MACD(異同移動平均線)"):
    fig4 = make_subplots(specs=[[{"secondary_y": True}]])
    
    # #### include candlestick with rangeselector
    # fig4.add_trace(go.Candlestick(x=KBar_df['Time'],
    #                 open=KBar_df['Open'], high=KBar_df['High'],
    #                 low=KBar_df['Low'], close=KBar_df['Close'], name='K線'),
    #                secondary_y=True)   ## secondary_y=True 表示此圖形的y軸scale是在右邊而不是在左邊
    
    #### include a go.Bar trace for volumes
    fig4.add_trace(go.Bar(x=KBar_df['time'], y=KBar_df['MACD_Histogram'], name='MACD Histogram', marker=dict(color='black')),secondary_y=False)  ## secondary_y=False 表示此圖形的y軸scale是在左邊而不是在右邊
    fig4.add_trace(go.Scatter(x=KBar_df['time'][last_nan_index_MACD+1:], y=KBar_df['Signal_Line'][last_nan_index_MACD+1:], mode='lines',line=dict(color='orange', width=2), name='訊號線(DEA)'), 
                  secondary_y=True)
    fig4.add_trace(go.Scatter(x=KBar_df['time'][last_nan_index_MACD+1:], y=KBar_df['MACD'][last_nan_index_MACD+1:], mode='lines',line=dict(color='pink', width=2), name='DIF'), 
                  secondary_y=True)
    
    fig4.layout.yaxis2.showgrid=True
    st.plotly_chart(fig4, use_container_width=True)



#%%
####### (6) 程式交易 #######
st.subheader("程式交易:")

###### 函數定義: 繪製K線圖加上MA以及下單點位
# @st.cache_data(ttl=3600, show_spinner="正在加載資料...")  ## Add the caching decorator
def ChartOrder_MA(Kbar_df,TR):
    # # 將K線轉為DataFrame
    # Kbar_df=KbarToDf(KBar)
    # 買(多)方下單點位紀錄
    BTR = [ i for i in TR if i[0]=='Buy' or i[0]=='B' ]
    BuyOrderPoint_date = [] 
    BuyOrderPoint_price = []
    BuyCoverPoint_date = []
    BuyCoverPoint_price = []
    for date,Low,High in zip(Kbar_df['time'],Kbar_df['low'],Kbar_df['high']):
        # 買方進場
        if date in [ i[2] for i in BTR ]:
            BuyOrderPoint_date.append(date)
            BuyOrderPoint_price.append(Low * 0.999)
        else:
            BuyOrderPoint_date.append(np.nan)
            BuyOrderPoint_price.append(np.nan)
        # 買方出場
        if date in [ i[4] for i in BTR ]:
            BuyCoverPoint_date.append(date)
            BuyCoverPoint_price.append(High * 1.001)
        else:
            BuyCoverPoint_date.append(np.nan)
            BuyCoverPoint_price.append(np.nan)
    # # 將下單點位加入副圖物件
    # if [ i for i in BuyOrderPoint if not np.isnan(i) ] !=[]:
    #     addp.append(mpf.make_addplot(BuyOrderPoint,scatter=True,markersize=50,marker='^',color='red'))  ## 200
    #     addp.append(mpf.make_addplot(BuyCoverPoint,scatter=True,markersize=50,marker='v',color='blue')) ## 200
    # 賣(空)方下單點位紀錄
    STR = [ i for i in TR if i[0]=='Sell' or i[0]=='S' ]
    SellOrderPoint_date = []
    SellOrderPoint_price = []
    SellCoverPoint_date = []
    SellCoverPoint_price = []
    for date,Low,High in zip(Kbar_df['time'],Kbar_df['low'],Kbar_df['high']):
        # 賣方進場
        if date in [ i[2] for i in STR]:
            SellOrderPoint_date.append(date)
            SellOrderPoint_price.append(High * 1.001)
        else:
            SellOrderPoint_date.append(np.nan)
            SellOrderPoint_price.append(np.nan)
        # 賣方出場
        if date in [ i[4] for i in STR ]:
            SellCoverPoint_date.append(date)
            SellCoverPoint_price.append(Low * 0.999)
        else:
            SellCoverPoint_date.append(np.nan)
            SellCoverPoint_price.append(np.nan)
    # # 將下單點位加入副圖物件
    # if [ i for i in SellOrderPoint if not np.isnan(i) ] !=[]:
    #     addp.append(mpf.make_addplot(SellOrderPoint,scatter=True,markersize=50,marker='v',color='green'))  ## 200
    #     addp.append(mpf.make_addplot(SellCoverPoint,scatter=True,markersize=50,marker='^',color='pink'))   ## 200
    # 開始繪圖
    # ChartKBar(KBar,addp,volume_enable)
    fig5 = make_subplots(specs=[[{"secondary_y": True}]])
    
    #### include candlestick with rangeselector
    # fig5.add_trace(go.Candlestick(x=KBar_df['time'],
    #                 open=KBar_df['open'], high=KBar_df['high'],
    #                 low=KBar_df['low'], close=KBar_df['close'], name='K線'),
    #                 secondary_y=False)   ## secondary_y=True 表示此圖形的y軸scale是在右邊而不是在左邊
    
    #### include a go.Bar trace for volumes
    # fig5.add_trace(go.Bar(x=KBar_df['time'], y=KBar_df['volume'], name='成交量', marker=dict(color='black')),secondary_y=False)  ## secondary_y=False 表示此圖形的y軸scale是在左邊而不是在右邊
    fig5.add_trace(go.Scatter(x=KBar_df['time'][last_nan_index_MA_trading+1:], y=KBar_df['MA_long'][last_nan_index_MA_trading+1:], mode='lines',line=dict(color='orange', width=2), name=f'{LongMAPeriod}-根 K棒 移動平均線'), 
                  secondary_y=False)
    fig5.add_trace(go.Scatter(x=KBar_df['time'][last_nan_index_MA_trading+1:], y=KBar_df['MA_short'][last_nan_index_MA_trading+1:], mode='lines',line=dict(color='pink', width=2), name=f'{ShortMAPeriod}-根 K棒 移動平均線'), 
                  secondary_y=False)
    fig5.add_trace(go.Scatter(x=BuyOrderPoint_date, y=BuyOrderPoint_price, mode='markers',  marker=dict(color='red', symbol='triangle-up', size=10),  name='作多進場點'), secondary_y=False)
    fig5.add_trace(go.Scatter(x=BuyCoverPoint_date, y=BuyCoverPoint_price, mode='markers',  marker=dict(color='blue', symbol='triangle-down', size=10),  name='作多出場點'), secondary_y=False)
    fig5.add_trace(go.Scatter(x=SellOrderPoint_date, y=SellOrderPoint_price, mode='markers',  marker=dict(color='green', symbol='triangle-down', size=10),  name='作空進場點'), secondary_y=False)
    fig5.add_trace(go.Scatter(x=SellCoverPoint_date, y=SellCoverPoint_price, mode='markers',  marker=dict(color='black', symbol='triangle-up', size=10),  name='作空出場點'), secondary_y=False)
 
    fig5.layout.yaxis2.showgrid=True
    st.plotly_chart(fig5, use_container_width=True)

###### 選擇不同交易策略:
choices_strategy = ['<進場>: 移動平均線黃金交叉作多,死亡交叉作空. <出場>: 結算平倉(期貨), 移動停損.','<進場>: RSI 超買超賣作多/空. <出場>: RSI 回歸中軸, 移動停損.',
    '<進場>: 布林通道突破邊緣作多/空. <出場>: 觸及中軌, 移動停損.' ]
choice_strategy = st.selectbox('選擇交易策略', choices_strategy, index=0)

##### 各別不同策略
if choice_strategy == '<進場>: 移動平均線黃金交叉作多,死亡交叉作空. <出場>: 結算平倉(期貨), 移動停損.':
    #### 選擇參數
    with st.expander("<策略參數設定>: 交易停損量、長移動平均線(MA)的K棒週期數目、短移動平均線(MA)的K棒週期數目、購買數量"):
        MoveStopLoss = st.slider('選擇程式交易停損量(股票:每股價格; 期貨(大小台指):台股指數點數. 例如: 股票進場做多時, 取30代表停損價格為目前每股價格減30元; 大小台指進場做多時, 取30代表停損指數為目前台股指數減30點)', 0, 100, 30, key='MoveStopLoss')
        LongMAPeriod_trading=st.slider('設定計算長移動平均線(MA)的 K棒週期數目(整數, 例如 10)', 0, 100, 10, key='trading_MA_long')
        ShortMAPeriod_trading=st.slider('設定計算短移動平均線(MA)的 K棒週期數目(整數, 例如 2)', 0, 100, 2, key='trading_MA_short')
        Order_Quantity = st.slider('選擇購買數量(股票單位為張數(一張為1000股); 期貨單位為口數)', 1, 100, 1, key='Order_Quantity')
    
        ### 計算長短移動平均線
        KBar_df['MA_long'] = Calculate_MA(KBar_df, period=LongMAPeriod_trading)
        KBar_df['MA_short'] = Calculate_MA(KBar_df, period=ShortMAPeriod_trading)
        
        ### 尋找最後 NAN值的位置
        last_nan_index_MA_trading = KBar_df['MA_long'][::-1].index[KBar_df['MA_long'][::-1].apply(pd.isna)][0]


        
        ### 建立部位管理物件
        OrderRecord=Record() 
        
        # ###### 變為字典
        # # KBar_dic = KBar_df_original.to_dict('list')
        # KBar_dic = KBar_df.to_dict('list')
        
    #### 開始回測
    for n in range(1,len(KBar_df['time'])-1):
        # 先判斷long MA的上一筆值是否為空值 再接續判斷策略內容
        if not np.isnan( KBar_df['MA_long'][n-1] ) :
            ## 進場: 如果無未平倉部位 
            if OrderRecord.GetOpenInterest()==0 :
                # 多單進場: 黃金交叉: short MA 向上突破 long MA
                if KBar_df['MA_short'][n-1] <= KBar_df['MA_long'][n-1] and KBar_df['MA_short'][n] > KBar_df['MA_long'][n] :
                    OrderRecord.Order('Buy', KBar_df['product'][n+1],KBar_df['time'][n+1],KBar_df['open'][n+1],Order_Quantity)
                    OrderPrice = KBar_df['open'][n+1]
                    StopLossPoint = OrderPrice - MoveStopLoss
                    continue
                # 空單進場:死亡交叉: short MA 向下突破 long MA
                if KBar_df['MA_short'][n-1] >= KBar_df['MA_long'][n-1] and KBar_df['MA_short'][n] < KBar_df['MA_long'][n] :
                    OrderRecord.Order('Sell', KBar_df['product'][n+1],KBar_df['time'][n+1],KBar_df['open'][n+1],Order_Quantity)
                    OrderPrice = KBar_df['open'][n+1]
                    StopLossPoint = OrderPrice + MoveStopLoss
                    continue
            # 多單出場: 如果有多單部位   
            elif OrderRecord.GetOpenInterest()>0 :
                ## 結算平倉(期貨才使用, 股票除非是下市櫃)
                if KBar_df['product'][n+1] != KBar_df['product'][n] :
                    OrderRecord.Cover('Sell', KBar_df['product'][n],KBar_df['time'][n],KBar_df['close'][n],OrderRecord.GetOpenInterest())
                    continue
                # 逐筆更新移動停損價位
                if KBar_df['close'][n] - MoveStopLoss > StopLossPoint :
                    StopLossPoint = KBar_df['close'][n] - MoveStopLoss
                # 如果上一根K的收盤價觸及停損價位，則在最新時間出場
                elif KBar_df['close'][n] < StopLossPoint :
                    OrderRecord.Cover('Sell', KBar_df['product'][n+1],KBar_df['time'][n+1],KBar_df['open'][n+1],OrderRecord.GetOpenInterest())
                    continue
            # 空單出場: 如果有空單部位
            elif OrderRecord.GetOpenInterest()<0 :
                ## 結算平倉(期貨才使用, 股票除非是下市櫃)
                if KBar_df['product'][n+1] != KBar_df['product'][n] :
               
                    OrderRecord.Cover('Buy', KBar_df['product'][n],KBar_df['time'][n],KBar_df['close'][n],-OrderRecord.GetOpenInterest())
                    continue
                # 逐筆更新移動停損價位
                if KBar_df['close'][n] + MoveStopLoss < StopLossPoint :
                    StopLossPoint = KBar_df['close'][n] + MoveStopLoss
                # 如果上一根K的收盤價觸及停損價位，則在最新時間出場
                elif KBar_df['close'][n] > StopLossPoint :
                    OrderRecord.Cover('Buy', KBar_df['product'][n+1],KBar_df['time'][n+1],KBar_df['open'][n+1],-OrderRecord.GetOpenInterest())
                    continue

    #### 繪製K線圖加上MA以及下單點位    
    ChartOrder_MA(KBar_df,OrderRecord.GetTradeRecord())
    
    
    
# 策略二：RSI 超買超賣策略
elif choice_strategy == choices_strategy[1]:
    st.subheader("RSI 策略參數設定:")
    with st.expander("<策略參數設定>: 交易停損量、RSI 週期、超買區間、超賣區間、購買數量"):
        MoveStopLoss_RSI = st.slider('選擇程式交易停損量', 0, 100, 30, key='MoveStopLoss_RSI')
        RSI_Period = st.slider('設定計算 RSI 的 K棒週期數目(整數, 例如 14)', 1, 30, 14, key='RSI_Period')
        RSI_Overbought = st.slider('設定 RSI 超買區間 (例如 70)', 50, 100, 70, key='RSI_Overbought')
        RSI_Oversold = st.slider('設定 RSI 超賣區間 (例如 30)', 0, 50, 30, key='RSI_Oversold')
        Order_Quantity_RSI = st.slider('選擇購買數量(股票單位為張數; 期貨單位為口數)', 1, 100, 1, key='Order_Quantity_RSI')

        # 確保超賣 < 超買
        if RSI_Oversold >= RSI_Overbought:
            st.error("RSI 超賣區間必須小於超買區間，請重新設定。")
            # 這裡可以選擇停止回測或使用預設值
            st.stop()


        ### 計算 RSI
        KBar_df = Calculate_RSI(KBar_df.copy(), period=RSI_Period) # .copy() 防止 SettingWithCopyWarning

        ### 尋找最後 NAN值的位置 (用於繪圖起始點)
        if not KBar_df['RSI'].isnull().all():
            last_nan_index_RSI_trading = KBar_df['RSI'][::-1].index[KBar_df['RSI'][::-1].isnull()][0]
        else:
            last_nan_index_RSI_trading = -1

        ### 建立部位管理物件
        OrderRecord_RSI = Record() # 使用不同的 Record 實例以避免混淆

    #### 開始回測
    # 注意：這裡使用 KBar_df，確保它包含了 RSI 欄位
    for n in range(1, len(KBar_df['time'])-1): # 迴圈範圍調整以避免索引錯誤
        # 先判斷 RSI 值是否為空值
        if not np.isnan(KBar_df['RSI'][n-1]):
            ## 進場: 如果無未平倉部位
            if OrderRecord_RSI.GetOpenInterest() == 0:
                # 多單進場: RSI 超賣區間 (從下方突破超賣線)
                if KBar_df['RSI'][n-1] <= RSI_Oversold and KBar_df['RSI'][n] > RSI_Oversold:
                    OrderRecord_RSI.Order('Buy', KBar_df['product'][n+1], KBar_df['time'][n+1], KBar_df['open'][n+1], Order_Quantity_RSI)
                    OrderPrice = KBar_df['open'][n+1]
                    StopLossPoint = OrderPrice - MoveStopLoss_RSI
                    continue
                # 空單進場: RSI 超買區間 (從上方跌破超買線)
                if KBar_df['RSI'][n-1] >= RSI_Overbought and KBar_df['RSI'][n] < RSI_Overbought:
                    OrderRecord_RSI.Order('Sell', KBar_df['product'][n+1], KBar_df['time'][n+1], KBar_df['open'][n+1], Order_Quantity_RSI)
                    OrderPrice = KBar_df['open'][n+1]
                    StopLossPoint = OrderPrice + MoveStopLoss_RSI
                    continue
            # 多單出場: 如果有多單部位
            elif OrderRecord_RSI.GetOpenInterest() > 0:
                ## 結算平倉(期貨才使用, 股票除非是下市櫃) - 同 MA 策略的平倉邏輯
                if KBar_df['product'][n+1] != KBar_df['product'][n]:
                    OrderRecord_RSI.Cover('Sell', KBar_df['product'][n], KBar_df['time'][n], KBar_df['close'][n], OrderRecord_RSI.GetOpenInterest())
                    continue
                # 移動停損
                if KBar_df['close'][n] - MoveStopLoss_RSI > StopLossPoint:
                    StopLossPoint = KBar_df['close'][n] - MoveStopLoss_RSI
                elif KBar_df['close'][n] < StopLossPoint:
                    OrderRecord_RSI.Cover('Sell', KBar_df['product'][n+1], KBar_df['time'][n+1], KBar_df['open'][n+1], OrderRecord_RSI.GetOpenInterest())
                    continue
                # RSI 回歸中軸平倉 (從超賣回到中線以上)
                if KBar_df['RSI'][n] > 50 and KBar_df['RSI'][n-1] <= 50: # 或者您可以設定成回到超賣區間上方即平倉，例如 KBar_df['RSI'][n] > RSI_Oversold
                    OrderRecord_RSI.Cover('Sell', KBar_df['product'][n+1], KBar_df['time'][n+1], KBar_df['open'][n+1], OrderRecord_RSI.GetOpenInterest())
                    continue
            # 空單出場: 如果有空單部位
            elif OrderRecord_RSI.GetOpenInterest() < 0:
                ## 結算平倉(期貨才使用, 股票除非是下市櫃) - 同 MA 策略的平倉邏輯
                if KBar_df['product'][n+1] != KBar_df['product'][n]:
                    OrderRecord_RSI.Cover('Buy', KBar_df['product'][n], KBar_df['time'][n], KBar_df['close'][n], -OrderRecord_RSI.GetOpenInterest())
                    continue
                # 移動停損
                if KBar_df['close'][n] + MoveStopLoss_RSI < StopLossPoint:
                    StopLossPoint = KBar_df['close'][n] + MoveStopLoss_RSI
                elif KBar_df['close'][n] > StopLossPoint:
                    OrderRecord_RSI.Cover('Buy', KBar_df['product'][n+1], KBar_df['time'][n+1], KBar_df['open'][n+1], -OrderRecord_RSI.GetOpenInterest())
                    continue
                # RSI 回歸中軸平倉 (從超買回到中線以下)
                if KBar_df['RSI'][n] < 50 and KBar_df['RSI'][n-1] >= 50: # 或者您可以設定成回到超買區間下方即平倉，例如 KBar_df['RSI'][n] < RSI_Overbought
                    OrderRecord_RSI.Cover('Buy', KBar_df['product'][n+1], KBar_df['time'][n+1], KBar_df['open'][n+1], -OrderRecord_RSI.GetOpenInterest())
                    continue

    # 處理最後未平倉部位 (如果回測結束時還有部位)
    if OrderRecord_RSI.GetOpenInterest() != 0:
        last_index = len(KBar_df['time']) - 1
        last_product = KBar_df['product'][last_index]
        last_time = KBar_df['time'][last_index]
        last_close = KBar_df['close'][last_index]
        if OrderRecord_RSI.GetOpenInterest() > 0:
            OrderRecord_RSI.Cover('Sell', last_product, last_time, last_close, OrderRecord_RSI.GetOpenInterest())
        else:
            OrderRecord_RSI.Cover('Buy', last_product, last_time, last_close, -OrderRecord_RSI.GetOpenInterest())

    #### 繪製K線圖加上RSI以及下單點位 (您需要為RSI策略調整繪圖函數，或者建立一個新的)
    # 這裡我先用 ChartOrder_MA 作為 placeholder，但實際您可能需要客製化一個 ChartOrder_RSI
    st.write("RSI 策略的交易圖示:")
    ChartOrder_MA(KBar_df, OrderRecord_RSI.GetTradeRecord()) # 臨時使用MA的繪圖函數
    
# 策略三：布林通道逆勢策略 
elif choice_strategy == choices_strategy[2]:
    st.subheader("布林通道逆勢策略參數設定:")
    with st.expander("<策略參數設定>: 交易停損量、布林通道週期、標準差倍數、購買數量"):
        MoveStopLoss_BB = st.slider('選擇程式交易停損量', 0, 100, 30, key='MoveStopLoss_BB')
        BB_Period = st.slider('設定布林通道的 K棒週期數目(整數, 例如 20)', 5, 50, 20, key='BB_Period')
        BB_NumStdDev = st.slider('設定布林通道標準差倍數(例如 2.0)', 1.0, 3.0, 2.0, 0.1, key='BB_NumStdDev')
        Order_Quantity_BB = st.slider('選擇購買數量(股票單位為張數; 期貨單位為口數)', 1, 100, 1, key='Order_Quantity_BB')

        # 計算布林通道
        KBar_df = Calculate_Bollinger_Bands(KBar_df.copy(), period=BB_Period, num_std_dev=BB_NumStdDev)

        # 檢查是否有足夠的數據計算布林通道
        if KBar_df['MiddleBand'].isnull().all():
            st.warning("數據不足以計算布林通道，請檢查週期設定或K棒數量。")
            st.stop() # 停止程式執行，避免報錯

        ### 建立部位管理物件
        OrderRecord_BB = Record() # 使用不同的 Record 實例

    #### 開始回測
    for n in range(1, len(KBar_df['time'])-1): # 迴圈範圍
        # 確保布林通道值存在
        if not np.isnan(KBar_df['MiddleBand'][n-1]):
            ## 進場: 如果無未平倉部位
            if OrderRecord_BB.GetOpenInterest() == 0:
                # 多單進場: 收盤價跌破下軌 (超賣反彈)
                if KBar_df['close'][n] < KBar_df['LowerBand'][n]:
                    OrderRecord_BB.Order('Buy', KBar_df['product'][n+1], KBar_df['time'][n+1], KBar_df['open'][n+1], Order_Quantity_BB)
                    OrderPrice = KBar_df['open'][n+1]
                    StopLossPoint = OrderPrice - MoveStopLoss_BB
                    continue
                # 空單進場: 收盤價突破上軌 (超買回落)
                if KBar_df['close'][n] > KBar_df['UpperBand'][n]:
                    OrderRecord_BB.Order('Sell', KBar_df['product'][n+1], KBar_df['time'][n+1], KBar_df['open'][n+1], Order_Quantity_BB)
                    OrderPrice = KBar_df['open'][n+1]
                    StopLossPoint = OrderPrice + MoveStopLoss_BB
                    continue
            # 多單出場: 如果有多單部位
            elif OrderRecord_BB.GetOpenInterest() > 0:
                # 結算平倉 (期貨)
                if KBar_df['product'][n+1] != KBar_df['product'][n]:
                    OrderRecord_BB.Cover('Sell', KBar_df['product'][n], KBar_df['time'][n], KBar_df['close'][n], OrderRecord_BB.GetOpenInterest())
                    continue
                # 移動停損
                if KBar_df['close'][n] - MoveStopLoss_BB > StopLossPoint:
                    StopLossPoint = KBar_df['close'][n] - MoveStopLoss_BB
                elif KBar_df['close'][n] < StopLossPoint:
                    OrderRecord_BB.Cover('Sell', KBar_df['product'][n+1], KBar_df['time'][n+1], KBar_df['open'][n+1], OrderRecord_BB.GetOpenInterest())
                    continue
                # 觸及中軌平倉 (多單獲利了結或止損)
                if KBar_df['close'][n] > KBar_df['MiddleBand'][n] and KBar_df['close'][n-1] <= KBar_df['MiddleBand'][n-1]: # 向上穿越中軌
                    OrderRecord_BB.Cover('Sell', KBar_df['product'][n+1], KBar_df['time'][n+1], KBar_df['open'][n+1], OrderRecord_BB.GetOpenInterest())
                    continue
            # 空單出場: 如果有空單部位
            elif OrderRecord_BB.GetOpenInterest() < 0:
                # 結算平倉 (期貨)
                if KBar_df['product'][n+1] != KBar_df['product'][n]:
                    OrderRecord_BB.Cover('Buy', KBar_df['product'][n], KBar_df['time'][n], KBar_df['close'][n], -OrderRecord_BB.GetOpenInterest())
                    continue
                # 移動停損
                if KBar_df['close'][n] + MoveStopLoss_BB < StopLossPoint:
                    StopLossPoint = KBar_df['close'][n] + MoveStopLoss_BB
                elif KBar_df['close'][n] > StopLossPoint:
                    OrderRecord_BB.Cover('Buy', KBar_df['product'][n+1], KBar_df['time'][n+1], KBar_df['open'][n+1], -OrderRecord_BB.GetOpenInterest())
                    continue
                # 觸及中軌平倉 (空單獲利了結或止損)
                if KBar_df['close'][n] < KBar_df['MiddleBand'][n] and KBar_df['close'][n-1] >= KBar_df['MiddleBand'][n-1]: # 向下穿越中軌
                    OrderRecord_BB.Cover('Buy', KBar_df['product'][n+1], KBar_df['time'][n+1], KBar_df['open'][n+1], -OrderRecord_BB.GetOpenInterest())
                    continue

    # 處理最後未平倉部位
    if OrderRecord_BB.GetOpenInterest() != 0:
        last_index = len(KBar_df['time']) - 1
        last_product = KBar_df['product'][last_index]
        last_time = KBar_df['time'][last_index]
        last_close = KBar_df['close'][last_index]
        if OrderRecord_BB.GetOpenInterest() > 0:
            OrderRecord_BB.Cover('Sell', last_product, last_time, last_close, OrderRecord_BB.GetOpenInterest())
        else:
            OrderRecord_BB.Cover('Buy', last_product, last_time, last_close, -OrderRecord_BB.GetOpenInterest())

    # 繪製 K 線圖和布林通道
    st.write("布林通道策略的交易圖示:")
    ChartOrder_MA(KBar_df, OrderRecord_BB.GetTradeRecord()) # ChartOrder_MA 已更新可繪製布林通道

##### 繪製K線圖加上MA以及下單點位
# @st.cache_data(ttl=3600, show_spinner="正在加載資料...")  ## Add the caching decorator
# def ChartOrder_MA(Kbar_df,TR):
#     # # 將K線轉為DataFrame
#     # Kbar_df=KbarToDf(KBar)
#     # 買(多)方下單點位紀錄
#     BTR = [ i for i in TR if i[0]=='Buy' or i[0]=='B' ]
#     BuyOrderPoint_date = [] 
#     BuyOrderPoint_price = []
#     BuyCoverPoint_date = []
#     BuyCoverPoint_price = []
#     for date,Low,High in zip(Kbar_df['time'],Kbar_df['low'],Kbar_df['high']):
#         # 買方進場
#         if date in [ i[2] for i in BTR ]:
#             BuyOrderPoint_date.append(date)
#             BuyOrderPoint_price.append(Low * 0.999)
#         else:
#             BuyOrderPoint_date.append(np.nan)
#             BuyOrderPoint_price.append(np.nan)
#         # 買方出場
#         if date in [ i[4] for i in BTR ]:
#             BuyCoverPoint_date.append(date)
#             BuyCoverPoint_price.append(High * 1.001)
#         else:
#             BuyCoverPoint_date.append(np.nan)
#             BuyCoverPoint_price.append(np.nan)
#     # # 將下單點位加入副圖物件
#     # if [ i for i in BuyOrderPoint if not np.isnan(i) ] !=[]:
#     #     addp.append(mpf.make_addplot(BuyOrderPoint,scatter=True,markersize=50,marker='^',color='red'))  ## 200
#     #     addp.append(mpf.make_addplot(BuyCoverPoint,scatter=True,markersize=50,marker='v',color='blue')) ## 200
#     # 賣(空)方下單點位紀錄
#     STR = [ i for i in TR if i[0]=='Sell' or i[0]=='S' ]
#     SellOrderPoint_date = []
#     SellOrderPoint_price = []
#     SellCoverPoint_date = []
#     SellCoverPoint_price = []
#     for date,Low,High in zip(Kbar_df['time'],Kbar_df['low'],Kbar_df['high']):
#         # 賣方進場
#         if date in [ i[2] for i in STR]:
#             SellOrderPoint_date.append(date)
#             SellOrderPoint_price.append(High * 1.001)
#         else:
#             SellOrderPoint_date.append(np.nan)
#             SellOrderPoint_price.append(np.nan)
#         # 賣方出場
#         if date in [ i[4] for i in STR ]:
#             SellCoverPoint_date.append(date)
#             SellCoverPoint_price.append(Low * 0.999)
#         else:
#             SellCoverPoint_date.append(np.nan)
#             SellCoverPoint_price.append(np.nan)
#     # # 將下單點位加入副圖物件
#     # if [ i for i in SellOrderPoint if not np.isnan(i) ] !=[]:
#     #     addp.append(mpf.make_addplot(SellOrderPoint,scatter=True,markersize=50,marker='v',color='green'))  ## 200
#     #     addp.append(mpf.make_addplot(SellCoverPoint,scatter=True,markersize=50,marker='^',color='pink'))   ## 200
#     # 開始繪圖
#     # ChartKBar(KBar,addp,volume_enable)
#     fig5 = make_subplots(specs=[[{"secondary_y": True}]])
    
#     #### include candlestick with rangeselector
#     # fig5.add_trace(go.Candlestick(x=KBar_df['time'],
#     #                 open=KBar_df['open'], high=KBar_df['high'],
#     #                 low=KBar_df['low'], close=KBar_df['close'], name='K線'),
#     #                 secondary_y=False)   ## secondary_y=True 表示此圖形的y軸scale是在右邊而不是在左邊
    
#     #### include a go.Bar trace for volumes
#     # fig5.add_trace(go.Bar(x=KBar_df['time'], y=KBar_df['volume'], name='成交量', marker=dict(color='black')),secondary_y=False)  ## secondary_y=False 表示此圖形的y軸scale是在左邊而不是在右邊
#     fig5.add_trace(go.Scatter(x=KBar_df['time'][last_nan_index_MA_trading+1:], y=KBar_df['MA_long'][last_nan_index_MA_trading+1:], mode='lines',line=dict(color='orange', width=2), name=f'{LongMAPeriod}-根 K棒 移動平均線'), 
#                   secondary_y=False)
#     fig5.add_trace(go.Scatter(x=KBar_df['time'][last_nan_index_MA_trading+1:], y=KBar_df['MA_short'][last_nan_index_MA_trading+1:], mode='lines',line=dict(color='pink', width=2), name=f'{ShortMAPeriod}-根 K棒 移動平均線'), 
#                   secondary_y=False)
#     fig5.add_trace(go.Scatter(x=BuyOrderPoint_date, y=BuyOrderPoint_price, mode='markers',  marker=dict(color='red', symbol='triangle-up', size=10),  name='作多進場點'), secondary_y=False)
#     fig5.add_trace(go.Scatter(x=BuyCoverPoint_date, y=BuyCoverPoint_price, mode='markers',  marker=dict(color='blue', symbol='triangle-down', size=10),  name='作多出場點'), secondary_y=False)
#     fig5.add_trace(go.Scatter(x=SellOrderPoint_date, y=SellOrderPoint_price, mode='markers',  marker=dict(color='green', symbol='triangle-down', size=10),  name='作空進場點'), secondary_y=False)
#     fig5.add_trace(go.Scatter(x=SellCoverPoint_date, y=SellCoverPoint_price, mode='markers',  marker=dict(color='black', symbol='triangle-up', size=10),  name='作空出場點'), secondary_y=False)
 
#     fig5.layout.yaxis2.showgrid=True
#     st.plotly_chart(fig5, use_container_width=True)


# ChartOrder_MA(KBar_df,OrderRecord.GetTradeRecord())






###### 計算績效:
# OrderRecord.GetTradeRecord()          ## 交易紀錄清單
# OrderRecord.GetProfit()               ## 利潤清單


def 計算績效_股票():
    交易總盈虧 = OrderRecord.GetTotalProfit()*1000          ## 取得交易總盈虧
    平均每次盈虧 = OrderRecord.GetAverageProfit()*1000         ## 取得交易 "平均" 盈虧(每次)
    平均投資報酬率 = OrderRecord.GetAverageProfitRate()    ## 取得交易 "平均" 投資報酬率(每次)  
    平均獲利_只看獲利的 = OrderRecord.GetAverEarn()*1000              ## 平均獲利(只看獲利的) 
    平均虧損_只看虧損的 = OrderRecord.GetAverLoss()*1000              ## 平均虧損(只看虧損的)
    勝率 = OrderRecord.GetWinRate()              ## 勝率
    最大連續虧損 = OrderRecord.GetAccLoss()*1000               ## 最大連續虧損
    最大盈虧回落_MDD = OrderRecord.GetMDD()*1000                   ## 最大利潤(盈虧)回落(MDD). 這個不是一般的 "資金" 或 "投資報酬率" 的回落
    if 最大盈虧回落_MDD>0:
        報酬風險比 = 交易總盈虧/最大盈虧回落_MDD
    else:
        報酬風險比='資料不足無法計算'
    return 交易總盈虧,平均每次盈虧,平均投資報酬率,平均獲利_只看獲利的,平均虧損_只看虧損的,勝率,最大連續虧損,最大盈虧回落_MDD,報酬風險比


def 計算績效_大台指期貨():
    交易總盈虧 = OrderRecord.GetTotalProfit()*200          ## 取得交易總盈虧
    平均每次盈虧 = OrderRecord.GetAverageProfit()*200         ## 取得交易 "平均" 盈虧(每次)
    平均投資報酬率 = OrderRecord.GetAverageProfitRate()    ## 取得交易 "平均" 投資報酬率(每次)  
    平均獲利_只看獲利的 = OrderRecord.GetAverEarn()*200              ## 平均獲利(只看獲利的) 
    平均虧損_只看虧損的 = OrderRecord.GetAverLoss()*200              ## 平均虧損(只看虧損的)
    勝率 = OrderRecord.GetWinRate()              ## 勝率
    最大連續虧損 = OrderRecord.GetAccLoss()*200               ## 最大連續虧損
    最大盈虧回落_MDD = OrderRecord.GetMDD()*200                   ## 最大利潤(盈虧)回落(MDD). 這個不是一般的 "資金" 或 "投資報酬率" 的回落
    if 最大盈虧回落_MDD>0:
        報酬風險比 = 交易總盈虧/最大盈虧回落_MDD
    else:
        報酬風險比='資料不足無法計算'
    return 交易總盈虧,平均每次盈虧,平均投資報酬率,平均獲利_只看獲利的,平均虧損_只看虧損的,勝率,最大連續虧損,最大盈虧回落_MDD,報酬風險比



def 計算績效_小台指期貨():
    交易總盈虧 = OrderRecord.GetTotalProfit()*50          ## 取得交易總盈虧
    平均每次盈虧 = OrderRecord.GetAverageProfit()*50         ## 取得交易 "平均" 盈虧(每次)
    平均投資報酬率 = OrderRecord.GetAverageProfitRate()    ## 取得交易 "平均" 投資報酬率(每次)  
    平均獲利_只看獲利的 = OrderRecord.GetAverEarn()*50              ## 平均獲利(只看獲利的) 
    平均虧損_只看虧損的 = OrderRecord.GetAverLoss()*50              ## 平均虧損(只看虧損的)
    勝率 = OrderRecord.GetWinRate()              ## 勝率
    最大連續虧損 = OrderRecord.GetAccLoss()*50               ## 最大連續虧損
    最大盈虧回落_MDD = OrderRecord.GetMDD()*50                   ## 最大利潤(盈虧)回落(MDD). 這個不是一般的 "資金" 或 "投資報酬率" 的回落
    if 最大盈虧回落_MDD>0:
        報酬風險比 = 交易總盈虧/最大盈虧回落_MDD
    else:
        報酬風險比='資料不足無法計算'
    return 交易總盈虧,平均每次盈虧,平均投資報酬率,平均獲利_只看獲利的,平均虧損_只看虧損的,勝率,最大連續虧損,最大盈虧回落_MDD,報酬風險比


def 計算績效_期貨():
    交易總盈虧 = OrderRecord.GetTotalProfit()*2        ## 取得交易總盈虧
    平均每次盈虧 = OrderRecord.GetAverageProfit()*2         ## 取得交易 "平均" 盈虧(每次)
    平均投資報酬率 = OrderRecord.GetAverageProfitRate()    ## 取得交易 "平均" 投資報酬率(每次)  
    平均獲利_只看獲利的 = OrderRecord.GetAverEarn()*2              ## 平均獲利(只看獲利的) 
    平均虧損_只看虧損的 = OrderRecord.GetAverLoss()*2              ## 平均虧損(只看虧損的)
    勝率 = OrderRecord.GetWinRate()              ## 勝率
    最大連續虧損 = OrderRecord.GetAccLoss()*2               ## 最大連續虧損
    最大盈虧回落_MDD = OrderRecord.GetMDD()*2                  ## 最大利潤(盈虧)回落(MDD). 這個不是一般的 "資金" 或 "投資報酬率" 的回落
    if 最大盈虧回落_MDD>0:
        報酬風險比 = 交易總盈虧/最大盈虧回落_MDD
    else:
        報酬風險比='資料不足無法計算'
    return 交易總盈虧,平均每次盈虧,平均投資報酬率,平均獲利_只看獲利的,平均虧損_只看虧損的,勝率,最大連續虧損,最大盈虧回落_MDD,報酬風險比





if choice == '中鋼期貨: 2023.4.17 至 2025.4.17':
    交易總盈虧,平均每次盈虧,平均投資報酬率,平均獲利_只看獲利的,平均虧損_只看虧損的,勝率,最大連續虧損,最大盈虧回落_MDD,報酬風險比 = 計算績效_期貨()
    # 交易總盈虧 = OrderRecord.GetTotalProfit()*1000          ## 取得交易總盈虧
    # 平均每次盈虧 = OrderRecord.GetAverageProfit()*1000         ## 取得交易 "平均" 盈虧(每次)
    # 平均投資報酬率 = OrderRecord.GetAverageProfitRate()    ## 取得交易 "平均" 投資報酬率(每次)  
    # 平均獲利_只看獲利的 = OrderRecord.GetAverEarn()*1000              ## 平均獲利(只看獲利的) 
    # 平均虧損_只看虧損的 = OrderRecord.GetAverLoss()*1000              ## 平均虧損(只看虧損的)
    # 勝率 = OrderRecord.GetWinRate()              ## 勝率
    # 最大連續虧損 = OrderRecord.GetAccLoss()*1000               ## 最大連續虧損
    # 最大盈虧回落_MDD = OrderRecord.GetMDD()*1000                   ## 最大利潤(盈虧)回落(MDD). 這個不是一般的 "資金" 或 "投資報酬率" 的回落
    # if 最大盈虧回落_MDD>0:
    #     報酬風險比 = 交易總盈虧/最大盈虧回落_MDD
    # else:
    #     報酬風險比='資料不足無法計算'

if choice == '聯電期貨: 2023.4.17 至 2025.4.17':
    交易總盈虧,平均每次盈虧,平均投資報酬率,平均獲利_只看獲利的,平均虧損_只看虧損的,勝率,最大連續虧損,最大盈虧回落_MDD,報酬風險比 = 計算績效_期貨()

    # 交易總盈虧 = OrderRecord.GetTotalProfit()*200          ## 取得交易總盈虧
    # 平均每次盈虧 = OrderRecord.GetAverageProfit() *200       ## 取得交易 "平均" 盈虧(每次)
    # 平均投資報酬率 = OrderRecord.GetAverageProfitRate()    ## 取得交易 "平均" 投資報酬率(每次)  
    # 平均獲利_只看獲利的 = OrderRecord.GetAverEarn() *200            ## 平均獲利(只看獲利的) 
    # 平均虧損_只看虧損的 = OrderRecord.GetAverLoss()*200             ## 平均虧損(只看虧損的)
    # 勝率 = OrderRecord.GetWinRate()              ## 勝率
    # 最大連續虧損 = OrderRecord.GetAccLoss()*200              ## 最大連續虧損
    # 最大盈虧回落_MDD = OrderRecord.GetMDD()*200                  ## 最大利潤(盈虧)回落(MDD). 這個不是一般的 "資金" 或 "投資報酬率" 的回落
    # if 最大盈虧回落_MDD>0:
    #     報酬風險比 = 交易總盈虧/最大盈虧回落_MDD
    # else:
    #     報酬風險比='資料不足無法計算'

if choice == '台積電期貨: 2023.4.17 至 2025.4.17':
    交易總盈虧,平均每次盈虧,平均投資報酬率,平均獲利_只看獲利的,平均虧損_只看虧損的,勝率,最大連續虧損,最大盈虧回落_MDD,報酬風險比 = 計算績效_期貨()
    # 交易總盈虧 = OrderRecord.GetTotalProfit()*50          ## 取得交易總盈虧
    # 平均每次盈虧 = OrderRecord.GetAverageProfit() *50       ## 取得交易 "平均" 盈虧(每次)
    # 平均投資報酬率 = OrderRecord.GetAverageProfitRate()    ## 取得交易 "平均" 投資報酬率(每次)  
    # 平均獲利_只看獲利的 = OrderRecord.GetAverEarn() *50            ## 平均獲利(只看獲利的) 
    # 平均虧損_只看虧損的 = OrderRecord.GetAverLoss()*50             ## 平均虧損(只看虧損的)
    # 勝率 = OrderRecord.GetWinRate()              ## 勝率
    # 最大連續虧損 = OrderRecord.GetAccLoss()*50              ## 最大連續虧損
    # 最大盈虧回落_MDD = OrderRecord.GetMDD()*50                  ## 最大利潤(盈虧)回落(MDD). 這個不是一般的 "資金" 或 "投資報酬率" 的回落
    # if 最大盈虧回落_MDD>0:
    #     報酬風險比 = 交易總盈虧/最大盈虧回落_MDD
    # else:
    #     報酬風險比='資料不足無法計算'

if choice == '富邦金期貨: 2023.4.17 至 2025.4.17':
    交易總盈虧,平均每次盈虧,平均投資報酬率,平均獲利_只看獲利的,平均虧損_只看虧損的,勝率,最大連續虧損,最大盈虧回落_MDD,報酬風險比 = 計算績效_期貨()
    # 交易總盈虧 = OrderRecord.GetTotalProfit()*1000          ## 取得交易總盈虧
    # 平均每次盈虧 = OrderRecord.GetAverageProfit()*1000         ## 取得交易 "平均" 盈虧(每次)
    # 平均投資報酬率 = OrderRecord.GetAverageProfitRate()    ## 取得交易 "平均" 投資報酬率(每次)  
    # 平均獲利_只看獲利的 = OrderRecord.GetAverEarn()*1000              ## 平均獲利(只看獲利的) 
    # 平均虧損_只看虧損的 = OrderRecord.GetAverLoss()*1000              ## 平均虧損(只看虧損的)
    # 勝率 = OrderRecord.GetWinRate()              ## 勝率
    # 最大連續虧損 = OrderRecord.GetAccLoss()*1000               ## 最大連續虧損
    # 最大盈虧回落_MDD = OrderRecord.GetMDD()*1000                   ## 最大利潤(盈虧)回落(MDD). 這個不是一般的 "資金" 或 "投資報酬率" 的回落
    # if 最大盈虧回落_MDD>0:
    #     報酬風險比 = 交易總盈虧/最大盈虧回落_MDD
    # else:
    #     報酬風險比='資料不足無法計算'

if choice == '台新金期貨: 2023.4.17 至 2025.4.17':
    交易總盈虧,平均每次盈虧,平均投資報酬率,平均獲利_只看獲利的,平均虧損_只看虧損的,勝率,最大連續虧損,最大盈虧回落_MDD,報酬風險比 = 計算績效_期貨()
if choice == '統一期貨: 2023.4.17至 2025.4.17 ':
    交易總盈虧,平均每次盈虧,平均投資報酬率,平均獲利_只看獲利的,平均虧損_只看虧損的,勝率,最大連續虧損,最大盈虧回落_MDD,報酬風險比 = 計算績效_期貨()
if choice == '金融期貨: 2023.4.17 至 2025.4.17 ':
    交易總盈虧,平均每次盈虧,平均投資報酬率,平均獲利_只看獲利的,平均虧損_只看虧損的,勝率,最大連續虧損,最大盈虧回落_MDD,報酬風險比 = 計算績效_大台指期貨()
if choice == '小型臺指期: 2023.4.17至 202.4.17 ':
    交易總盈虧,平均每次盈虧,平均投資報酬率,平均獲利_只看獲利的,平均虧損_只看虧損的,勝率,最大連續虧損,最大盈虧回落_MDD,報酬風險比 = 計算績效_小台指期貨()
if choice == '臺股期貨: 2023.4.17至 2025.4.17 ':
    交易總盈虧,平均每次盈虧,平均投資報酬率,平均獲利_只看獲利的,平均虧損_只看虧損的,勝率,最大連續虧損,最大盈虧回落_MDD,報酬風險比 = 計算績效_大台指期貨()
if choice == '元大台灣50: 2023.4.17至 2025.4.17 ':
    交易總盈虧,平均每次盈虧,平均投資報酬率,平均獲利_只看獲利的,平均虧損_只看虧損的,勝率,最大連續虧損,最大盈虧回落_MDD,報酬風險比 = 計算績效_股票()
if choice == '元大台灣50正2: 2023.4.17至 2025.4.17 ':
    交易總盈虧,平均每次盈虧,平均投資報酬率,平均獲利_只看獲利的,平均虧損_只看虧損的,勝率,最大連續虧損,最大盈虧回落_MDD,報酬風險比 = 計算績效_股票()
if choice == '台積電: 2023.4.17至 2025.4.17 ':
    交易總盈虧,平均每次盈虧,平均投資報酬率,平均獲利_只看獲利的,平均虧損_只看虧損的,勝率,最大連續虧損,最大盈虧回落_MDD,報酬風險比 = 計算績效_股票()
if choice == '華碩: 2023.4.17 至 2025.4.17':
    交易總盈虧,平均每次盈虧,平均投資報酬率,平均獲利_只看獲利的,平均虧損_只看虧損的,勝率,最大連續虧損,最大盈虧回落_MDD,報酬風險比 = 計算績效_股票()

# OrderRecord.GetCumulativeProfit()         ## 累計盈虧
# OrderRecord.GetCumulativeProfit_rate()    ## 累計投資報酬率

##### 将投資績效存储成一个DataFrame並以表格形式呈現各項績效數據
if len(OrderRecord.Profit)>0:
    data = {
        "項目": ["交易總盈虧(元)", "平均每次盈虧(元)", "平均投資報酬率", "平均獲利(只看獲利的)(元)", "平均虧損(只看虧損的)(元)", "勝率", "最大連續虧損(元)", "最大盈虧回落(MDD)(元)", "報酬風險比(交易總盈虧/最大盈虧回落(MDD))"],
        "數值": [交易總盈虧, 平均每次盈虧, 平均投資報酬率, 平均獲利_只看獲利的, 平均虧損_只看虧損的, 勝率, 最大連續虧損, 最大盈虧回落_MDD, 報酬風險比]
    }
    df = pd.DataFrame(data)
    if len(df)>0:
        st.write(df)
else:
    st.write('沒有交易記錄(已經了結之交易) !')






# ###### 累計盈虧 & 累計投資報酬率
# with st.expander("累計盈虧 & 累計投資報酬率"):
#     fig4 = make_subplots(specs=[[{"secondary_y": True}]])
    
#     #### include a go.Bar trace for volumes
#     # fig4.add_trace(go.Bar(x=KBar_df['Time'], y=KBar_df['MACD_Histogram'], name='MACD Histogram', marker=dict(color='black')),secondary_y=False)  ## secondary_y=False 表示此圖形的y軸scale是在左邊而不是在右邊
#     fig4.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_MACD+1:], y=KBar_df['Signal_Line'][last_nan_index_MACD+1:], mode='lines',line=dict(color='orange', width=2), name='訊號線(DEA)'), 
#                   secondary_y=True)
#     fig4.add_trace(go.Scatter(x=KBar_df['Time'][last_nan_index_MACD+1:], y=KBar_df['MACD'][last_nan_index_MACD+1:], mode='lines',line=dict(color='pink', width=2), name='DIF'), 
#                   secondary_y=True)
    
#     fig4.layout.yaxis2.showgrid=True
#     st.plotly_chart(fig4, use_container_width=True)



# #### 定義圖表
# matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'
# matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# ax1 = plt.subplot(2,1,1)
# ax2 = plt.subplot(2,1,2)




##### 畫累計盈虧圖:
if choice ==choices[0]  :  ##'中鋼期貨: 2023.4.17 至 2025.4.17'
    OrderRecord.GeneratorProfitChart(choice='future',StrategyName='MA')
if choice ==choices[1]  : ##'聯電期貨: 2023.4.17 至 2025.4.17':
    OrderRecord.GeneratorProfitChart(choice='future',StrategyName='MA')
if choice == choices[2]  :##'台積電期貨: 2023.4.17 至 2025.4.17':
    OrderRecord.GeneratorProfitChart(choice='future',StrategyName='MA')
if choice ==choices[3]  : ##'富邦金期貨: 2023.4.17 至 2025.4.17':
    OrderRecord.GeneratorProfitChart(choice='future',StrategyName='MA')
if choice ==choices[4]  : ##'台新金期貨: 2023.4.17 至 2025.4.17':
    OrderRecord.GeneratorProfitChart(choice='future',StrategyName='MA')
if choice ==choices[5]  : ##'統一期貨: 2023.4.17 至 2025.4.17':
    OrderRecord.GeneratorProfitChart(choice='future',StrategyName='MA')
if choice ==choices[6]  : ##'金融期貨: 2023.4.17 至 2025.4.17':
    OrderRecord.GeneratorProfitChart(choice='future',StrategyName='MA')
if choice ==choices[7]  : ##'小型臺指期: 2023.4.17 至 2025.4.17':
    OrderRecord.GeneratorProfitChart(choice='future',StrategyName='MA')
if choice ==choices[8]  : ##'臺股期貨: 2023.4.17 至 2025.4.17':
    OrderRecord.GeneratorProfitChart(choice='future',StrategyName='MA')
if choice ==choices[9]  : ##'元大台灣50: 2023.4.17 至 2025.4.17':
    OrderRecord.GeneratorProfitChart(choice='stock',StrategyName='MA')
if choice ==choices[10]  :## '元大台灣50正2: 2023.4.17 至 2025.4.17':
    OrderRecord.GeneratorProfitChart(choice='stock',StrategyName='MA')
if choice ==choices[11]  : ##'台積電: 2023.4.17 至 2025.4.17':
    OrderRecord.GeneratorProfitChart(choice='stock',StrategyName='MA')
if choice ==choices[12]  :## '華碩: 2023.4.17 至 2025.4.17':
    OrderRecord.GeneratorProfitChart(choice='stock',StrategyName='MA')
    

# matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'
# matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# plt.figure()

# #### 計算累計績效
# TotalProfit=[0]
# for i in OrderRecord.Profit:
#     TotalProfit.append(TotalProfit[-1]+i)

# #### 繪製圖形
# if choice == '台積電: 2022.1.1 至 2024.4.9':
#     # ax.plot( TotalProfit[1:]  , '-', marker='o', linewidth=1 )
#     plt.plot( TotalProfit[1:]*1000  , '-', marker='o', linewidth=1 )
# if choice == '大台指2024.12到期: 2024.1 至 2024.4.9':
#     # ax.plot( TotalProfit[1:]  , '-', marker='o', linewidth=1 )
#     plt.plot( TotalProfit[1:]*200  , '-', marker='o', linewidth=1 )


# ####定義標頭
# # # ax.set_title('Profit')
# # ax.set_title('累計盈虧')
# # ax.set_xlabel('交易編號')
# # ax.set_ylabel('累計盈虧(元/每股)')
# plt.title('累計盈虧(元)')
# plt.xlabel('交易編號')
# plt.ylabel('累計盈虧(元)')
# # if choice == '台積電: 2022.1.1 至 2024.4.9':
# #     plt.ylabel('累計盈虧(元/每股)')
# # if choice == '大台指2024.12到期: 2024.1 至 2024.4.9':
# #     plt.ylabel('累計盈虧(元/每口)')

# #### 设置x轴的刻度
# ### 获取TotalProfit的长度
# length = len(TotalProfit)
# ### 创建新的x轴刻度列表，每个值都加1
# new_ticks = range(1, length + 1)
# ### 应用新的x轴刻度
# plt.xticks(ticks=range(length), labels=new_ticks)

# #### 顯示繪製圖表
# # plt.show()    # 顯示繪製圖表
# # plt.savefig(StrategyName+'.png') #儲存繪製圖表
# ### 在Streamlit中显示
# st.pyplot(plt)






##### 畫累計投資報酬率圖:
OrderRecord.GeneratorProfit_rateChart(StrategyName='MA')
# matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'
# matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# plt.figure()

# #### 計算累計計投資報酬
# TotalProfit_rate=[0]
# for i in OrderRecord.Profit_rate:
#     TotalProfit_rate.append(TotalProfit_rate[-1]+i)

# #### 繪製圖形
# plt.plot( TotalProfit_rate[1:]  , '-', marker='o', linewidth=1 )
# # if choice == '台積電: 2022.1.1 至 2024.4.9':
# #     # ax.plot( TotalProfit[1:]  , '-', marker='o', linewidth=1 )
# #     plt.plot( TotalProfit_rate[1:]  , '-', marker='o', linewidth=1 )
# # if choice == '大台指2024.12到期: 2024.1 至 2024.4.9':
# #     # ax.plot( TotalProfit[1:]  , '-', marker='o', linewidth=1 )
# #     plt.plot( TotalProfit_rate[1:]  , '-', marker='o', linewidth=1 )


# ####定義標頭
# plt.title('累計投資報酬率')
# plt.xlabel('交易編號')
# plt.ylabel('累計投資報酬率')
# # if choice == '台積電: 2022.1.1 至 2024.4.9':
# #     plt.ylabel('累計投資報酬率')
# # if choice == '大台指2024.12到期: 2024.1 至 2024.4.9':
# #     plt.ylabel('累計投資報酬率')

# #### 设置x轴的刻度
# ### 获取TotalProfit的长度
# length = len(TotalProfit_rate)
# ### 创建新的x轴刻度列表，每个值都加1
# new_ticks = range(1, length + 1)
# ### 应用新的x轴刻度
# plt.xticks(ticks=range(length), labels=new_ticks)

# #### 顯示繪製圖表
# # plt.show()    # 顯示繪製圖表
# # plt.savefig(StrategyName+'.png') #儲存繪製圖表
# ### 在Streamlit中显示
# st.pyplot(plt)


#%%
####### (7) 呈現即時資料 #######






