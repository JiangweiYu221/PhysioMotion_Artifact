"""
Project Name:
Author: Jiangwei Yu
Contact: yjiangwei210@163.com
Last Updated: 2025-03-19
Copyright (c) 2025 Jiangwei Yu
Licensed under the MIT License
Project URL:
Description:
This code provides a comprehensive labeling tool for preprocessed data,
leveraging the powerful plotting capabilities of Python-MNE.
It enables experts to:
1. Select Specific Data Components:
    Easily choose subjects, run numbers, channels and review the data before labeling.
2. Interactive Data Labeling:
    Utilize an interactive annotation function implemented by Python-MNE to label the data efficiently.
3. Automated Label Storage:
    Once the labeling process is complete, the labels are automatically saved to a CSV file.

"""


import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QGridLayout, QCheckBox, QPushButton, QDialog,
    QLabel, QLineEdit, QFormLayout, QComboBox, QMessageBox
)
from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtGui import QFont
import pandas as pd
import os
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mne_bids import BIDSPath, read_raw_bids




matplotlib.use("Qt5Agg")
bids_root = ""  # root path of preprocessed data
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("选择列表")
        self.resize(400, 600)
        # main window Layout
        self.layout = QGridLayout(self)
        # font setting
        self.font = QFont("Arial", 12)
        # comboBox
        self.combo_box = QComboBox()
        self.combo_box.setFont(self.font)
        self.combo_box.addItems(["close_base", "open_base", "ver_eyem", "hor_eyem", "blink", "hor_headm",
                                 "ver_headm", "tongue", "chew", "swallow", "eyebrow", "blink_hor_headm",
                                 "blink_ver_headm", "blink_eyebrow", "tongue_eyebrow", "swallow_eyebrow"])

        # adding comboBox to main window
        self.layout.addWidget(self.combo_box,2, 0, 1, 2, Qt.AlignCenter)
        self.combo_box.setCurrentIndex(-1)
        self.combo_box.currentIndexChanged.connect(self.setcheckbox)
        self.combo_box.setMinimumSize(120, 40)

        # memory function
        settings = QSettings("labeling", "Artifact")
        saved_subject = settings.value("subject", "")
        saved_run = settings.value("run", "")
        self.textboxfont = QFont("Arial", 15)

        # textboxes for subject ID and run number
        self.textbox1 = QLineEdit()
        self.textbox1.setFont(self.textboxfont)
        self.textbox1.setPlaceholderText("subject")
        self.textbox1.setText(saved_subject)

        self.textbox2 = QLineEdit()
        self.textbox2.setFont(self.textboxfont)
        self.textbox2.setPlaceholderText("run")
        self.textbox2.setText(saved_run)

        self.layout.addWidget(self.textbox1, 0, 0, 1, 2, Qt.AlignCenter)
        self.layout.addWidget(self.textbox2, 1, 0, 1, 2, Qt.AlignCenter)



        # checkboxes for channels
        self.elements = [
            'Fp1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
            'Fz-Cz', 'Cz-Pz', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fp2-F8', 'F8-T8',
            'T8-P8', 'P8-O2', 'F7-Fp1', 'Fp1-Fp2', 'F7-F3', 'F3-Fz', 'Fz-F4', 'F4-F8',
            'T7-C3', 'C3-Cz', 'Cz-C4', 'C4-T8', 'P7-P3', 'P3-Pz', 'Pz-P4', 'P4-P8',
            'O1-O2', 'O2-P8', 'ALL'
        ]
        self.checkboxes = []
        for idx, element in enumerate(self.elements):
            checkbox = QCheckBox(element)
            checkbox.setObjectName(element)
            checkbox.setFont(self.font)
            self.checkboxes.append(checkbox)
            row = 3 + idx % 18
            col = idx // 18
            self.layout.addWidget(checkbox, row, col)



        # confirm button connected to labeling function
        self.confirm_button = QPushButton("确认")
        self.confirm_button.setFont(self.font)
        self.confirm_button.clicked.connect(self.open_new_window)
        self.layout.addWidget(self.confirm_button, len(self.elements) // 2 + 5, 1,1,1, Qt.AlignCenter)

        # review button connected to reviewing data
        self.choose_button = QPushButton("选择通道")
        self.choose_button.setFont(self.font)
        self.choose_button.clicked.connect(self.open_mne)
        self.layout.addWidget(self.choose_button, len(self.elements) // 2 + 5, 0,1,1, Qt.AlignCenter)


    def setcheckbox(self):
        label = self.combo_box.currentText()

        # empirical preset channels corresponding to different artifacts
        csv_file = "sample.csv"
        df = pd.read_csv(csv_file)
        filtered = df[df['label'] == label]
        select = filtered['channel'].drop_duplicates().tolist()
        for checkbox in self.checkboxes:
            objname = checkbox.objectName()
            if objname in select:
                checkbox.setChecked(True)
            else:
                checkbox.setChecked(False)
        self.update()


    def open_new_window(self):
        self.selected_items = [cb.text() for cb in self.checkboxes if cb.isChecked()]
        if self.selected_items:
            self.dialog = InputDialog(self.selected_items, parent=self)
            self.dialog.show()

    def open_mne(self):
        self.subject_name = self.textbox1.text()

        #BIDS path format
        session = 'preprocessed'
        task = 'artifact'
        datatype = 'eeg'
        self.subject_run = self.textbox2.text()
        bids_path = BIDSPath(subject=self.subject_name, session=session, task=task,
                             run=int(self.subject_run), root=bids_root, datatype=datatype)
        self.data = read_raw_bids(bids_path)
        self.figure = Figure(figsize=(1, 1))
        self.canvas = FigureCanvas(self.figure)
        matplotlib.rcParams['xtick.labelsize'] = 18
        matplotlib.rcParams['ytick.labelsize'] = 12

        # temporary txt file storing annotations
        annot_file = "sub" + self.subject_name + "_run0" + self.subject_run + "_annotations.txt"
        start_time = 0
        if os.path.isfile(annot_file):
            with open(annot_file, 'r') as file:
                lines = file.readlines()
                if lines:
                    start, duration, _ = lines[-1].strip().split(',')
                    start_time = float(start) + float(duration)
        self.data.plot(duration=40, n_channels=34, scalings=120, start=start_time, clipping=10)
        # update canvas
        self.canvas.draw()


class InputDialog(QDialog):
    def __init__(self, selected_items, parent=None):
        super().__init__(parent)
        self.selected_items = selected_items
        self.setWindowTitle("选择时间")
        self.resize(600, 400)

        # create layout
        layout = QFormLayout(self)

        # font setting
        self.font = QFont("Arial", 12)
        label_font = QFont("Arial", 16)

        # reminding text of chosen channels
        self.label = QLabel(f"选中的通道:\n {', '.join(selected_items)}")
        self.label.setFont(label_font)
        self.label.setWordWrap(True)
        layout.addWidget(self.label)

        # reminding text of chosen subject ID
        label_hint_sub = QLabel("subject:" + self.parent().textbox1.text())
        label_hint_sub.setFont(label_font)
        layout.addWidget(label_hint_sub)

        # reminding text of chosen run number
        label_hint_sub = QLabel("run:" + self.parent().textbox2.text())
        label_hint_sub.setFont(label_font)
        layout.addWidget(label_hint_sub)

        # reminding text of current label
        label_hint_sub = QLabel("label:" + self.parent().combo_box.currentText())
        label_hint_sub.setFont(label_font)
        layout.addWidget(label_hint_sub)

        # button to call python-mne
        self.button_font = QFont("Arial", 20)
        self.plot_button = QPushButton("调用MNE")
        self.plot_button.setFont(self.button_font)
        self.plot_button.clicked.connect(self.open_plot_window)
        self.plot_button.setMinimumSize(150, 100)
        layout.addWidget(self.plot_button)

        # button to save csv file
        self.confirm_button = QPushButton("保存")
        self.confirm_button.setFont(self.button_font)
        self.confirm_button.clicked.connect(self.save_data)
        self.confirm_button.setMinimumSize(150, 100)
        layout.addWidget(self.confirm_button)

        self.center_window()
        self.setLayout(layout)

    def open_plot_window(self):
        self.subject_name = self.parent().textbox1.text()

        # BIDS path format
        session = 'preprocessed'
        task = 'artifact'
        datatype = 'eeg'
        self.subject_run = self.parent().textbox2.text()
        bids_path = BIDSPath(subject=self.subject_name, session=session, task=task,
                             run=int(self.subject_run), root=bids_root, datatype=datatype)
        self.data = read_raw_bids(bids_path)
        self.figure = Figure(figsize=(1, 1))
        self.canvas = FigureCanvas(self.figure)
        matplotlib.rcParams['xtick.labelsize'] = 18
        matplotlib.rcParams['ytick.labelsize'] = 12

        # temporary txt file storing annotations
        annot_file = "sub" + self.subject_name + "_run0" + self.subject_run + "_annotations.txt"
        start_time = 0
        if os.path.isfile(annot_file):
            with open(annot_file, 'r') as file:
                lines = file.readlines()
                if lines:
                    start, duration ,_  = lines[-1].strip().split(',')
                    start_time = float(start) + float(duration)
        self.data.plot(duration=40, n_channels=34, scalings=120, start = start_time, clipping = 10)

        # update canvas
        self.canvas.draw()



    def center_window(self):
        if self.parent():
            parent_geometry = self.parent().geometry()
            parent_center = parent_geometry.center()
            self_geometry = self.frameGeometry()
            self_geometry.moveCenter(parent_center)
            self.move(self_geometry.topLeft())

    def save_data(self):
        selected_channels = self.selected_items
        label = self.parent().combo_box.currentText()

        # memory function
        settings = QSettings("labeling","Artifact")
        settings.setValue("subject", self.subject_name)
        settings.setValue("run", self.subject_run)

        # converting temporary txt file to csv file
        annot_file = "sub" + self.subject_name + "_run0" + self.subject_run + "_annotations.txt"
        annotations = self.data.annotations
        selected_annotations = [
            (onset, duration, description)
            for onset, duration, description in zip(annotations.onset, annotations.duration,
                                                    annotations.description)
            if description == 'BAD_'
        ]
        with open(annot_file, 'w') as f:
            for onset, duration, description in selected_annotations:
                f.write(f'{onset},{duration},{description}\n')

        start_time = []
        stop_time = []
        with open(annot_file, 'r') as file:
            for line in file:
                if line.strip().startswith('#') or not line.strip():
                    continue
                parts = line.strip().split(',')
                onset, duration, _ = parts
                start_time.append(onset)
                stop_time.append((float(onset) + float(duration)))

        # creating dataframe
        data = {
            "channel": [],
            "start_time": [],
            "stop_time": [],
            "label": []
        }

        for i in range(len(start_time)):
            for channel in selected_channels:
                data["channel"].append(channel)
                data["start_time"].append(start_time[i])
                data["stop_time"].append(stop_time[i])
                data["label"].append(label)

        df = pd.DataFrame(data)
        self.file_path = "sub" + self.subject_name + "_run0" + self.subject_run + ".csv"
        if not os.path.isfile(self.file_path):
            df.to_csv(self.file_path, index=False)
            self.show_create_message()
            self.accept()
        else:
            df.to_csv(self.file_path, mode="a", header=False, index=False)
            self.show_success_message()
            for checkbox in self.parent().checkboxes:
                checkbox.setChecked(False)
            self.accept()

    def show_success_message(self):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("保存成功")
        msg.setText(f"文件已保存到: {self.file_path}")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def show_create_message(self):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("创建成功")
        msg.setText("csv文件创建成功")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()


def run_labeling():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

run_labeling()


