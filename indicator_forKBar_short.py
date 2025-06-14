# 載入必要套件
import requests, datetime, os, time
import numpy as np
import matplotlib.dates as mdates
# from talib.abstract import * # 載入技術指標函數

# 算K棒
class KBar():
    # 設定初始化變數
    def __init__(self, date, cycle=1):
        # K棒的頻率(分鐘)
        self.TAKBar = {}
        self.TAKBar['time'] = np.array([])
        self.TAKBar['open'] = np.array([])
        self.TAKBar['high'] = np.array([])
        self.TAKBar['low'] = np.array([])
        self.TAKBar['close'] = np.array([])
        self.TAKBar['volume'] = np.array([])

        # --- 修正開始 ---
        # 確保 self.current 是 datetime 物件
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

        self.cycle = datetime.timedelta(minutes=cycle) # K棒週期 (以分鐘計)

    # 更新最新報價
    def AddPrice(self, time, open_price, close_price, low_price, high_price, volume):
        # 同一根K棒
        if time <= self.current:
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
        # 不同根K棒
        else:
            while time > self.current:
                self.current += self.cycle
            self.TAKBar['time'] = np.append(self.TAKBar['time'], self.current)
            self.TAKBar['open'] = np.append(self.TAKBar['open'], open_price)
            self.TAKBar['high'] = np.append(self.TAKBar['high'], high_price)
            self.TAKBar['low'] = np.append(self.TAKBar['low'], low_price)
            self.TAKBar['close'] = np.append(self.TAKBar['close'], close_price)
            self.TAKBar['volume'] = np.append(self.TAKBar['volume'], volume)
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

    # 以下被註解掉的技術指標函數，如果你要使用 TALIB 相關功能，記得解除註解並確保已安裝 TALIB 函式庫
    # # 取MA值(MA期數)
    # def GetMA(self,n,matype):
    #       return MA(self.TAKBar,n,matype)
    # # 取SMA值(SMA期數)
    # def GetSMA(self,n):
    #       return SMA(self.TAKBar,n)
    # # 取WMA值(WMA期數)
    # def GetWMA(self,n):
    #       return WMA(self.TAKBar,n)
    # # 取EMA值(EMA期數)
    # def GetEMA(self,n):
    #       return EMA(self.TAKBar,n)
    # # 取布林通道值(中線期數)
    # def GetBBands(self,n):
    #       return BBANDS(self.TAKBar,n) ##BBANDS()函數有很多選項,此處只使用期數 n
    # # RSI(RSI期數)
    # def GetRSI(self,n):
    #       return RSI(self.TAKBar,n)
    # # 取KD值(RSV期數,K值期數,D值期數)
    # def GetKD(self,rsv,k,d):
    #       return STOCH(self.TAKBar,fastk_period = rsv,slowk_period = k,slowd_period = d)
    # # 取得威廉指標
    # def GetWILLR(self,tp=14):
    #       return WILLR(self.TAKBar, timeperiod=tp)
    # # 取得乖離率
    # def GetBIAS(self,tn=10):
    #       mavalue=MA(self.TAKBar,timeperiod=tn,matype=0)
    #       return (self.TAKBar['close']-mavalue)/mavalue


            
