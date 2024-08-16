# Copy from the sample code (NoGuiGetData.py),provided by the organizer.

from KKT_Module.ksoc_global import kgl
from KKT_Module.Configs import SettingConfigs
from KKT_Module.SettingProcess.SettingProccess import SettingProc, ConnectDevice, ResetDevice
from KKT_Module.DataReceive.DataReciever import RawDataReceiver, HWResultReceiver, FeatureMapReceiver
import time

# 測試版子的輸入有沒有問題
def test_input(num):
    print(num + 1)
    
# 連接毫米波AI裝置
def connect():
    connect = ConnectDevice()
    connect.startUp()                       # Connect to the device
    reset = ResetDevice()
    reset.startUp()                         # Reset hardware register

# 導入毫米波AI裝置之參數    
def startSetting():
    SettingConfigs.setScriptDir("K60168-Test-00256-008-v0.0.8-20230717_120cm")
    ksp = SettingProc()                 # Object for setting process to setup the Hardware AI and RF before receive data
    ksp.startUp(SettingConfigs)         # Start the setting process
    
# 主程式迴圈
def startLoop():
    # kgl.ksoclib.switchLogMode(True)
    
    # ! parameter : chirps is chirps number.
    R = RawDataReceiver(chirps=32)

    # Receiver for getting Raw data
    # R = FeatureMapReceiver(chirps=32)       # Receiver for getting RDI PHD map
    # R = HWResultReceiver()                  # Receiver for getting hardware results (gestures, Axes, exponential)
    # buffer = DataBuffer(100)                # Buffer for saving latest frames of data
    R.trigger(chirps=32)                      # Trigger receiver before getting the data
    time.sleep(0.5)
    print('# ======== Start getting gesture ===========')
    
    # 主要程式會放在while loop中
    while True:                             # loop for getting the data
        res = R.getResults()                # Get data from receiver
        if res is None:
            continue
        else:
            print('data = {}'.format(res))          # Print results
            test_input(1)
        
        time.sleep(0.1)
        '''
        Main program in here. example:
        '''
            
# 啟動程式 ->載入上面所寫的function
def main():
    kgl.setLib()

    # kgl.ksoclib.switchLogMode(True)

    connect()                               # First you have to connect to the device

    startSetting()                         # Second you have to set the setting configs

    startLoop()                            # Last you can continue to get the data in the loop

if __name__ == '__main__':
    main()
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                       _oo0oo_
#                      o8888888o
#                      88" . "88
#                      (| -_- |)
#                      0\  =  /0
#                    ___/`---'\___
#                  .' \\|     |# '.
#                 / \\|||  :  |||# \
#                / _||||| -:- |||||- \
#               |   | \\\  -  #/ |   |
#               | \_|  ''\---/''  |_/ |
#               \  .-\__  '-'  ___/-. /
#             ___'. .'  /--.--\  `. .'___
#          ."" '<  `.___\_<|>_/___.' >' "".
#         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
#         \  \ `_.   \_ __\ /__ _/   .-` /  /
#     =====`-.____`.___ \_____/___.-`___.-'=====
#                       `=---='
#
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  
#               佛祖保佑         永無BUG