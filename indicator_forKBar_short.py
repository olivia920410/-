# 載入必要套件
import requests, datetime, os, time
import numpy as np
import matplotlib.dates as mdates
#from talib.abstract import * # 載入技術指標函數

# 算K棒
class KBar():
    # 設定初始化變數
    def __init__(self, date, cycle = 1):
        # K棒的頻率(分鐘)
        self.TAKBar = {}
        self.TAKBar['time'] = np.array([])
        self.TAKBar['open'] = np.array([])
        self.TAKBar['high'] = np.array([])
        self.TAKBar['low'] = np.array([])
        self.TAKBar['close'] = np.array([])
        self.TAKBar['volume'] = np.array([])

        # --- KBar __init__ 方法的關鍵修正 ---
        if isinstance(date, datetime.datetime):
            # 如果 'date' 已經是 datetime 物件，直接使用它
            self.current = date
        elif isinstance(date, str):
            # 如果 'date' 是字串，嘗試解析它
            try:
                # 嘗試解析包含時間的完整格式
                self.current = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                # 如果完整格式失敗，嘗試解析只有日期的格式
                try:
                    self.current = datetime.datetime.strptime(date, '%Y-%m-%d')
                except ValueError:
                    # 如果兩種常見格式都無法解析，則拋出錯誤
                    raise ValueError(f"無法解析傳入的日期字串 '{date}'。請確保其格式為 'YYYY-MM-DD' 或 'YYYY-MM-DD HH:MM:SS'。")
        else:
            # 如果 'date' 既不是 datetime 物件也不是字串，則拋出型別錯誤
            raise TypeError(f"傳入 KBar 類別的 'date' 參數型別錯誤。預期為 datetime 物件或字串，但收到 {type(date)}。")
        # --- 修正結束 ---

        self.cycle = datetime.timedelta(minutes = cycle)
    
    # 更新最新報價
    def AddPrice(self,time, open_price, close_price, low_price, high_price,volume):
        # time 和 self.current 都應該是 datetime 物件，這段比較應該正常
        if time <= self.current:
            # 更新收盤價
            self.TAKBar['close'][-1] = close_price
            # 更新成交量
            self.TAKBar['volume'][-1] += volume  
            # 更新最高價
            self.TAKBar['high'][-1] = max(self.TAKBar['high'][-1],high_price)
            # 更新最低價
            self.TAKBar['low'][-1] = min(self.TAKBar['low'][-1],low_price)  
            # 若沒有更新K棒，則回傳0
            return 0
        # 不同根K棒
        else:
            while time > self.current:
                self.current += self.cycle
            self.TAKBar['time'] = np.append(self.TAKBar['time'],self.current)
            self.TAKBar['open'] = np.append(self.TAKBar['open'],open_price)
            self.TAKBar['high'] = np.append(self.TAKBar['high'],high_price)
            self.TAKBar['low'] = np.append(self.TAKBar['low'],low_price)
            self.TAKBar['close'] = np.append(self.TAKBar['close'],close_price)
            self.TAKBar['volume'] = np.append(self.TAKBar['volume'],volume)
            # 若有更新K棒，則回傳1
            return 1
    
    # 取時間 (以下略，與你提供的程式碼相同)
    def GetTime(self):
        return self.TAKBar['time']     
    def GetOpen(self):
        return self.TAKBar['open']
    def GetHigh(self):
        return self.TAKBar['high']
    def GetLow(self):
        return self.TAKBar['low']
    def GetClose(self):
        return self.TAKBar['close']
    def GetVolume(self):
        return self.TAKBar['volume']
    # ... (其他 GetXXX 方法和註解掉的 TA-Lib 函數) ...


            
