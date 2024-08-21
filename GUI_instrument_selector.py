import sys
import pygame.midi as pm
import random
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QComboBox
from PyQt5.QtCore import QThread, pyqtSignal

# MIDI播放器
pm.init()
player = pm.Output(0)  # 使用Microsoft GS合成器发声

# 播放跟禁音
def play_note_on(pitch=60, velocity=127):
    player.note_on(pitch, velocity)

def play_note_off(pitch=60, velocity=127):
    player.note_off(pitch, velocity)
    
# 模擬即時label的預測，並且可以隨時更換樂器
class LabelThread(QThread):
    new_label = pyqtSignal(int)

    def run(self):
        while True:
            label = random.choice(range(0, 10))  # 模擬隨機label
            self.new_label.emit(label)  # 送出預測結果
            self.msleep(250)  # 模擬模型預測的間隔時間

# 主視窗
class MidiSelector(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.label_thread = LabelThread()
        self.label_thread.new_label.connect(self.handle_label) # 對預測出label產生對應行為
        self.label_thread.start()  # 開始預測

    def initUI(self):
        self.setWindowTitle('MIDI Instrument Selector')

        layout = QVBoxLayout()

        # 樂器選擇(文字)
        self.label = QLabel('Select an instrument:', self)
        layout.addWidget(self.label)

        # 下拉選項(樂器)
        self.combo = QComboBox(self)
        self.combo.addItem("Piano")
        self.combo.addItem("Guitar")
        self.combo.addItem("Violin")
        self.combo.addItem("Trumpet")
        layout.addWidget(self.combo)

        # 初始設置為鋼琴
        self.combo.currentIndexChanged.connect(self.change_instrument)
        self.set_instrument(0)

        self.setLayout(layout)

        # 初始音符參數
        self.notes = [60, 62, 64, 65, 67, 69, 71]
        self.Velo = 100
        self.last_note = self.notes[0]

    def set_instrument(self, index):
        instruments = {'Piano': 0, 'Guitar': 24, 'Violin': 40, 'Trumpet': 56}
        instrument_name = self.combo.currentText()
        player.set_instrument(instruments[instrument_name])

    def change_instrument(self):
        self.set_instrument(self.combo.currentIndex())

    def handle_label(self, label):
        print(f'Label: {label}', end=" ")
        if label in range(1, 8):  # 音符區間
            current = self.notes[label - 1]
            if current != self.last_note:  # 音符有變
                play_note_off(self.last_note, self.Velo)  # 禁音上一個音符
                print(f'Sent note_off: {self.last_note}', end=" ")
            play_note_on(pitch=current, velocity=self.Velo)  # 彈奏當前label對應的音符
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
    main()

# 關閉MIDI播放器
player.close()
pm.quit()

# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⠟⠛⠉⣩⣍⠉⠛⠻⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⠋⠀⠀⣠⣾⣿⠟⠁⠀⠀⠀⠙⣿⣿⣿⣿
# ⣿⣿⣿⠁⠀⠀⢾⣿⣟⠁⠀⣠⣾⣷⣄⠀⠘⣿⣿⣿
# ⣿⣿⡇⣠⣦⡀⠀⠙⢿⣷⣾⡿⠋⠻⣿⣷⣄⢸⣿⣿
# ⣿⣿⡇⠙⢿⣿⣦⣠⣾⡿⢿⣷⣄⠀⠈⠻⠋⢸⣿⣿
# ⣿⣿⣿⡀⠀⠙⢿⡿⠋⠀⢀⣽⣿⡷⠀⠀⢠⣿⣿⣿
# ⣿⣿⣿⣿⣄⠀⠀⠀⢀⣴⣿⡿⠋⠀⠀⣠⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣦⣤⣀⣙⣋⣀⣤⣴⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# 有Bug就發動閃電戰