import sys
import pygame.midi as pm
import random
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QComboBox
from PyQt5.QtCore import QThread, pyqtSignal
import time

from CNN_Ver10 import get_model
from GetData import connect, startSetting
from KKT_Module.ksoc_global import kgl
from KKT_Module.DataReceive.DataReciever import RawDataReceiver, FeatureMapReceiver
import numpy as np
from einops import rearrange

# MIDI播放器
pm.init()
player = pm.Output(0)  # 使用Microsoft GS合成器发声

model = get_model()
model.build((None, 12, 32, 32, 2))
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["acc"])
model.load_weights("ModelSave/CNN.h5")

# 播放跟禁音


def play_note_on(pitch=60, velocity=127):
    player.note_on(pitch, velocity)


def play_note_off(pitch=60, velocity=127):
    player.note_off(pitch, velocity)

# 模擬即時label的預測，並且可以隨時更換樂器


class LabelThread(QThread):
    new_label = pyqtSignal(int)

    def run(self):
        R = FeatureMapReceiver(chirps=32)
        R.trigger(chirps=32)  # Trigger receiver before getting the data
        input_data = []
        arr = np.array(None)
        count = 0
        init = True
        time.sleep(0.5)
        while True:
            res = R.getResults()
            if res is None:
                continue
            print('data = {}'.format(res))          # Print results

            if count < 12:
                data = np.array(res)
                input_data.append(data)
                count += 1

            elif count == 12:
                arr = np.array(input_data)
                arr = rearrange(arr, '(b f) c w h  -> b f w h c', b=1)
                print(arr.shape)
                label = model.predict(arr)  # 模擬隨機label
                label = np.argmax(label[0])
                self.new_label.emit(label)  # 送出預測結果
                count += 1
                # self.msleep(250)  # 模擬模型預測的間隔時間

            else:
                arr = rearrange(arr, 'b f w h c  -> (b f) c w h')
                arr = np.delete(arr, 0, 0)
                data = rearrange(np.array(res), '(f c) w h -> f c w h', f=1)
                arr = np.append(arr, data, axis=0)
                arr = rearrange(arr, '(b f) c w h  -> b f w h c', b=1)
                print(arr.shape)
                label = model.predict(arr)  # 模擬隨機label
                label = np.argmax(label[0])
                self.new_label.emit(label)  # 送出預測結果
                # self.msleep(250)  # 模擬模型預測的間隔時間

            # elif count % 12 <= 11:
            #     data = np.array(res)
            #     np.delete(arr, 0, 0)
            #     np.append(data)
            #     count += 1


# 主視窗
class MidiSelector(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()   
        self.label_thread = LabelThread()
        self.label_thread.new_label.connect(self.handle_label)  # 處理label
        self.label_thread.start()  # 開始預測label

    def initUI(self):
        # 初始音符参數
        self.notes = [60, 62, 64, 65, 67, 69, 71]
        self.Velo = 100
        self.last_note = self.notes[0]
        
        self.setStyleSheet("""QComboBox {
            border: 1px solid gray;
            border-radius: 3px;
            padding: 1px 18px 1px 3px;
            min-width: 6em;
        }

        QComboBox:editable {
            background: white;
        }

        QComboBox:!editable, QComboBox::drop-down:editable {
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                        stop: 0 #E1E1E1, stop: 0.4 #DDDDDD,
                                        stop: 0.5 #D8D8D8, stop: 1.0 #D3D3D3);
        }

        /* QComboBox gets the "on" state when the popup is open */
        QComboBox:!editable:on, QComboBox::drop-down:editable:on {
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                        stop: 0 #D3D3D3, stop: 0.4 #D8D8D8,
                                        stop: 0.5 #DDDDDD, stop: 1.0 #E1E1E1);
        }

        QComboBox:on { /* shift the text when the popup opens */
            padding-top: 3px;
            padding-left: 4px;
        }

        QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 15px;

            border-left-width: 1px;
            border-left-color: darkgray;
            border-left-style: solid; /* just a single line */
            border-top-right-radius: 3px; /* same radius as the QComboBox */
            border-bottom-right-radius: 3px;
        }

        QComboBox::down-arrow:on { /* shift the arrow when popup is open */
            top: 1px;
            left: 1px;
        }

        QComboBox QAbstractItemView {
            border: 2px solid darkgray;
            selection-background-color: lightgray;
        }
        """)
        self.setWindowTitle('MIDI Instrument Selector')
        layout = QVBoxLayout()
        self.resize(320,150)
        
        # 樂器選擇(文字)
        self.Text1 = QLabel('Select an instrument:', self)
        self.Text1.setGeometry(10,10,150,30)

        # 下拉菜單(樂器)
        self.selector_combo = QComboBox(self)
        self.selector_combo.setGeometry(10,45,300,30)

        # 音域選擇(文字)
        self.Text2 = QLabel('Select a range of notes:', self)
        self.Text2.setGeometry(10,75,150,30)
        
        #設置音域選擇下拉選單
        self.note_range_combo = QComboBox(self)
        self.note_range_combo.setGeometry(10,105,300,30)

        # 樂器分類以及對應的MIDI編號
        self.instruments = {
            'Piano': {'Acoustic Grand Piano': 0, 'Bright Acoustic Piano': 1},
            'Guitar': {'Acoustic Guitar (nylon)': 24, 'Electric Guitar (jazz)': 26},
            'Pipe': {'Church Organ': 19, 'Reed Organ': 20},
            'Brass': {'Trumpet': 56, 'Trombone': 57},
            'Ensemble': {'SynthStrings': 50, 'Choir Aahs': 52}
        }

        self.ranges = {'C2~B2': -24, 'C3~B3': -12, 'C4~B4': 0, 'C5~B5': 12}
        
        self.populate_instruments()  # 新增項目至樂器選擇下拉菜單
        self.populate_ranges()  #新增音域選擇至下拉菜單
        
        # 設置初始樂器及音域
        self.set_instrument(self.selector_combo.currentText())
        self.note_range_combo.setCurrentText('C4~B4')
        
        self.selector_combo.currentIndexChanged.connect(self.change_instrument) #變換樂器事件
        self.note_range_combo.currentIndexChanged.connect(self.change_range)    #變換音域事件

        self.setLayout(layout)


    def populate_instruments(self):
        self.selector_combo.clear()
        for category, instruments in self.instruments.items():
            self.selector_combo.addItem(f'--- {category} ---')  # 母標題(樂器分類，例如鋼琴類、吉他類等)
            for instrument in instruments:
                self.selector_combo.addItem(instrument)  # 在母標題下新增樂器

    def populate_ranges(self):
        self.note_range_combo.clear()
        for range in self.ranges:
            self.note_range_combo.addItem(range)  # 在母標題下新增樂器

    def set_instrument(self, instrument_name):
        for category, instruments in self.instruments.items():
            if instrument_name in instruments:
                player.set_instrument(instruments[instrument_name])
                break
    
    def set_range(self,range_name):
        for key, value in self.ranges.items():
            if range_name == key:
                self.notes = [note + value for note in [60, 62, 64, 65, 67, 69, 71]]
    
    def change_instrument(self):
        instrument_name = self.selector_combo.currentText()
        # 如果選擇的是母標題，不做任何操作
        if instrument_name.startswith('---'):
            return
        self.set_instrument(instrument_name)

    def change_range(self):
        range_name = self.note_range_combo.currentText()
        self.set_range(range_name)
        
    def handle_label(self, label):
        print(f'Label: {label}', end=" ")
        if label in range(1, 8):  # 音符區間
            current = self.notes[label - 1]
            if current != self.last_note:  # 音符有變
                play_note_off(self.last_note, self.Velo)  # 禁音上一音符
                print(f'Sent note_off: {self.last_note}', end=" ")
            play_note_on(pitch=current, velocity=self.Velo)  # 彈奏當前音符
            print(f'Playing note: {current} with velocity: {self.Velo}')
            self.last_note = current
        elif label == 8:  # 增加音量
            if self.Velo + 5 <= 127:
                self.Velo += 5
                print(f'Current velocity: {self.Velo}', end=" ")
                play_note_on(pitch=self.last_note, velocity=self.Velo)
                print(f'Playing note: {self.last_note} with velocity: {self.Velo}')
            else:
                print('Velocity must be less than 127')
        elif label == 9:  # 减小音量
            if self.Velo - 5 >= 0:
                self.Velo -= 5
                print(f'Current velocity: {self.Velo}', end=" ")
                play_note_on(pitch=self.last_note, velocity=self.Velo)
                print(f'Playing note: {self.last_note} with velocity: {self.Velo}')
            else:
                print('Velocity must be greater than 0')
        elif label == 0:  # 停止音符
            play_note_off(self.last_note, self.Velo)
            print(f'Sent note_off: {self.last_note}')
        else:
            print("sustain")


def main():
    app = QApplication(sys.argv)
    midi_selector = MidiSelector()
    midi_selector.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    kgl.setLib()
    connect()
    startSetting()
    main()

# 關閉MIDI播放器
player.close()
pm.quit()

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