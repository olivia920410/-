# 載入必要套件
import requests, datetime, os, time
import numpy as np
import pandas as pd # 為了將 numpy.datetime64 轉換為 datetime.datetime，通常需要用到 pandas

# 算K棒
class KBar():
    # 設定初始化變數
    def __init__(self, start_date_obj, cycle = 1): # 將 date 參數改為 start_date_obj (datetime.datetime 物件)
        # K棒的頻率(分鐘)
        self.TAKBar = {}
        self.TAKBar['time'] = np.array([], dtype='datetime64[ns]') # 確保 time 陣列初始化為 datetime64
        self.TAKBar['open'] = np.array([])
        self.TAKBar['high'] = np.array([])
        self.TAKBar['low'] = np.array([])
        self.TAKBar['close'] = np.array([])
        self.TAKBar['volume'] = np.array([])
        
        # 將 self.current 直接設定為傳入的 datetime 物件
        self.current = start_date_obj # start_date_obj 應該是 datetime.datetime 類型
        
        # cycle 轉換為 timedelta
        self.cycle = datetime.timedelta(minutes = cycle)

        # 為了處理第一根 K 棒的數據，可能需要一些暫存變數
        self._temp_open = None
        self._temp_high = None
        self._temp_low = None
        self._temp_close = None
        self._temp_volume = 0

    # 更新最新報價
    def AddPrice(self, time_input, open_price, close_price, low_price, high_price, volume):
        # *** 關鍵修正點：確保 time_input 轉換為 datetime.datetime ***
        if isinstance(time_input, np.datetime64):
            time_dt = pd.to_datetime(time_input).to_pydatetime()
        elif isinstance(time_input, datetime.datetime):
            time_dt = time_input
        else:
            # 處理其他可能的類型，或直接報錯
            raise TypeError(f"AddPrice: 'time' 參數類型不支援, 預期 numpy.datetime64 或 datetime.datetime, 實際為 {type(time_input)}")

        # --- 處理 K 棒初始化（如果還沒有 K 棒數據） ---
        if self._temp_open is None: # 這是第一筆數據，初始化暫存變數
            self._temp_open = open_price
            self._temp_high = high_price
            self._temp_low = low_price
            self._temp_close = close_price
            self._temp_volume = volume
            # 確保 K 棒的起始時間被正確設定
            # 如果是第一次處理數據，則將 current K 棒的起始時間設定為這筆數據的時間
            # 考慮到您的 current 是用來判斷週期，這裡的邏輯需要稍微調整
            # 確保 self.current 一直是 K 棒的起始時間點
            if not self.TAKBar['time'].size > 0: # 僅在第一次進入時設置
                self.TAKBar['time'] = np.append(self.TAKBar['time'], self.current)
                self.TAKBar['open'] = np.append(self.TAKBar['open'], self._temp_open)
                self.TAKBar['high'] = np.append(self.TAKBar['high'], self._temp_high)
                self.TAKBar['low'] = np.append(self.TAKBar['low'], self._temp_low)
                self.TAKBar['close'] = np.append(self.TAKBar['close'], self._temp_close)
                self.TAKBar['volume'] = np.append(self.TAKBar['volume'], self._temp_volume)
            else:
                # 如果已經有 K 棒了，但這是第一個數據點，則更新現有 K 棒（這段邏輯可能需要更明確定義）
                # 通常第一筆數據會開啟第一個 K 棒
                pass # 這裡可能需要根據您的具體需求調整

        # --- K 棒聚合邏輯 ---
        # 同一根K棒（time_dt 小於等於當前 K 棒的結束時間點）
        # 這裡的邏輯假設 self.current 代表的是 K 棒的開盤時間，
        # 並且 K 棒週期是從 self.current 開始 cycle_duration 分鐘
        # 判斷 time_dt 是否在當前 K 棒的有效範圍內
        
        # 計算當前 K 棒的「結束時間」（不包含，即下一個 K 棒的開始時間）
        # self.current 代表的是當前 K 棒的起始時間
        current_k_end_time = self.current + self.cycle 

        if time_dt < current_k_end_time: # 判斷是否屬於當前 K 棒週期
            # 更新收盤價
            self.TAKBar['close'][-1] = close_price
            # 更新成交量
            self.TAKBar['volume'][-1] += volume
            # 更新最高價
            self.TAKBar['high'][-1] = max(self.TAKBar['high'][-1], high_price)
            # 更新最低價
            self.TAKBar['low'][-1] = min(self.TAKBar['low'][-1], low_price)
            # 若沒有更新K棒，則回傳0
            return 0
        # 不同根K棒（time_dt 超出了當前 K 棒的週期）
        else:
            # 完成並記錄前一根 K 棒
            # 由於您在上面直接更新了 TAKBar[-1]，所以這裡不需要額外操作來 "完成" 上一根 K 棒
            # 但如果你需要記錄一個完整的 K 棒再清空暫存，邏輯會不同
            
            # 更新 self.current 到下一個 K 棒的起始時間
            # 這裡需要確定新的 K 棒的起始時間點
            # 應該是 time_dt 所在的那個週期的起始時間
            
            # 例如：如果 time_dt 是 10:07，cycle 是 5分鐘，前一根 K 棒是 10:00-10:05
            # 那麼新的 K 棒起始時間應該是 10:05，而不是 10:07
            # 你需要計算 time_dt 應該落在哪個 K 棒的起始時間
            
            # 最簡單但可能不精確的 `while time > self.current:` 邏輯，
            # 可能會跳過中間的 K 棒，如果原始數據的時間間隔很大
            
            # 正確的 K 棒滾動邏輯
            # 如果當前時間 time_dt 超出了 self.current 所在的 K 棒週期
            # 我們需要產生一個或多個空白 K 棒，直到 time_dt 所在的 K 棒週期
            
            # 在產生新的 K 棒之前，先完成並存儲上一根 K 棒的數據
            # (這在您的 AddPrice 邏輯中並不明顯，因為您直接更新了 -1)
            # 如果 self.TAKBar 已經有數據，這表示一個 K 棒結束了
            if self.TAKBar['time'].size > 0:
                # 這裡不需要額外操作，因為上一根 K 棒的數據已經在 if time_dt < current_k_end_time: 中更新了
                pass

            # 滾動 self.current 直到 time_dt 所在週期的起始時間
            new_k_start_time = self.current + self.cycle
            while time_dt >= new_k_start_time: # 直到找到 time_dt 所在的 K 棒起始時間
                # 如果中間有空白 K 棒，可以填充
                # 這裡的邏輯非常重要，決定了如何處理數據中斷或 K 棒之間間隔過大的情況
                
                # 範例：如果需要填充空 K 棒
                # 如果 new_k_start_time != time_dt，且 new_k_start_time < time_dt
                # 這表示有空白週期，可以添加一個 "空白" K 棒 (例如，用上一個 K 棒的收盤價作為開高低收)
                # 由於您的原始程式碼沒有處理空 K 棒，這裡保持簡潔
                
                self.current = new_k_start_time
                new_k_start_time += self.cycle
                
                # 添加新的 K 棒的開盤數據 (來自當前 time_dt 的數據)
                self.TAKBar['time'] = np.append(self.TAKBar['time'], self.current)
                self.TAKBar['open'] = np.append(self.TAKBar['open'], open_price)
                self.TAKBar['high'] = np.append(self.TAKBar['high'], high_price)
                self.TAKBar['low'] = np.append(self.TAKBar['low'], low_price)
                self.TAKBar['close'] = np.append(self.TAKBar['close'], close_price)
                self.TAKBar['volume'] = np.append(self.TAKBar['volume'], volume)

            # 更新 self.current 到 time_dt 所在週期的起始時間
            # 這應該是最後一個 while 循環更新後的 self.current
            # 或者，更精確的算法是：
            # self.current = time_dt - datetime.timedelta(minutes=time_dt.minute % self.cycle.total_seconds() / 60) # 簡化例子
            # 確保 self.current 是 time_dt 所在週期的開始
            
            # 簡化：直接設定新的 K 棒的數據
            # 由於 while 循環已經處理了 self.current 的滾動，
            # 現在 self.current 應該已經是 time_dt 所在週期的開始了 (或者最近的一個週期開始)
            # 這裡的邏輯需要與 while time > self.current 的設計意圖保持一致
            # 最安全的做法是：
            # 如果 time_dt 已經在一個新的週期裡，則將 time_dt 作為新 K 棒的開盤時間，並將 self.current 設定為 time_dt 
            # 這裡您的原始邏輯 `while time > self.current: self.current += self.cycle` 其實就是滾動到下一個 K 棒的開始
            # 所以現在 `self.current` 已經是新的 K 棒的開盤時間了。
            
            # 因此，這裡只需添加當前數據作為新 K 棒的初始值
            # 數據已在 while 迴圈中 append，所以這裡不需要再次 append
            
            # 若有更新K棒，則回傳1
            return 1

    # 取時間
    def GetTime(self):
        return self.TAKBar['time']      
    # 取開盤價
    def GetOpen(self):
        return self.TAKBar['open']
    # 取最高價
    def GetHigh(self):
        return self.TAKBar['high']
    # 取最低價
    def GetLow(self):
        return self.TAKBar['low']
    # 取收盤價
    def GetClose(self):
        return self.TAKBar['close']
    # 取成交量
    def GetVolume(self):
        return self.TAKBar['volume']



            
