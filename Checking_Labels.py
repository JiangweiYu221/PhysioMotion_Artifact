"""
Project Name:
Author: Jiangwei Yu
Contact: yjiangwei210@163.com
Last Updated: 2025-03-31
Copyright (c) 2025 Jiangwei Yu
Licensed under the MIT License
Project URL:
Description:
This code provides a plotting function for checking labels.
It distinguishes between task-related artifacts and task-unrelated signals.
Specifically, task-related artifacts are visually represented in red,
whereas task-unrelated signals are displayed in blue.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mne_bids import BIDSPath, read_raw_bids

subject_ID = ""
run_number = 1  #Integer datatype
label_path = ""
processed_data_path = ""

class EEGInteractivePlot:
    def __init__(self, bids_root, subject_ID, subject_run, csv_root):
        # read preprocessed data in BIDS format
        task = 'artifact'
        datatype = 'eeg'
        bids_path = BIDSPath(subject=subject_ID, task=task, run=subject_run,
                             root=bids_root, datatype=datatype)

        self.raw = read_raw_bids(bids_path)
        self.data, self.times = self.raw[:]
        self.channels = self.raw.ch_names[::-1]
        self.sfreq = self.raw.info['sfreq']
        self.n_rows = len(self.channels)

        # read auto marks
        tsv_path = BIDSPath(subject=subject_ID, task=task, run=subject_run,
                            root=bids_root, datatype='eeg', suffix='events', extension='tsv')
        self.events_df = pd.read_csv(tsv_path, sep='\t')

        # read csv manual labels
        csv_path = f"{csv_root}/sub{subject_ID}_run0{subject_run}.csv"
        self.df = pd.read_csv(csv_path)

        # initial parameters
        self.start_idx = 0
        self.duration = 40  # visible duration of the plotting window
        self.scale_factor = 0.15  # scaling
        self.max_idx = len(self.times) - int(self.duration * self.sfreq)

        # color setting
        self.color_no_label = 'blue'
        self.color_with_label = 'red'

        # global dy to fix the vertical distance between channels
        global_min, global_max = self.data.min(), self.data.max()
        self.dy = (global_max - global_min) * 0.15 * self.scale_factor

        # create the figure
        self.fig, self.axd = plt.subplot_mosaic([["EEG", "EEG"]], layout="constrained", figsize=(12, 6))
        self.ax_eeg = self.axd["EEG"]
        self.fig.subplots_adjust(bottom=0.25, top=0.95)

        # pre-create plotting objects
        self.lines = [self.ax_eeg.plot([], [], color=self.color_no_label,
                                       linewidth=0.2)[0] for _ in range(self.n_rows)]
        self.label_lines = [self.ax_eeg.plot([], [], color=self.color_with_label, linewidth=0.3)[0] for _ in
                            range(self.n_rows)]

        # pre-create auto marks line
        self.event_lines = [self.ax_eeg.axvline(0, color='green', linestyle='dashed',
                                                linewidth=3, alpha=0) for _ in
                            range(len(self.events_df))]

        # incorporate interactive controls
        self.add_widgets()
        self.plot_data()
        plt.show()

    def add_widgets(self):

        """incorporate interactive controls"""

        ax_prev = plt.axes([0.1, 0.1, 0.1, 0.04])
        ax_next = plt.axes([0.8, 0.1, 0.1, 0.04])
        ax_scale_up = plt.axes([0.4, 0.1, 0.1, 0.04])
        ax_scale_down = plt.axes([0.55, 0.1, 0.1, 0.04])
        ax_slider = plt.axes([0.25, 0.05, 0.5, 0.03])

        self.b_prev = Button(ax_prev, 'Previous')
        self.b_next = Button(ax_next, 'Next')
        self.b_scale_up = Button(ax_scale_up, '+ Scale')
        self.b_scale_down = Button(ax_scale_down, '- Scale')
        self.s_time = Slider(ax_slider, 'Time Window', 0,
                             self.max_idx / self.sfreq, valinit=0, valstep=30)

        self.b_prev.on_clicked(self.prev_window)
        self.b_next.on_clicked(self.next_window)
        self.b_scale_up.on_clicked(self.scale_up)
        self.b_scale_down.on_clicked(self.scale_down)
        self.s_time.on_changed(self.slider_update)

    def plot_data(self):

        """plot eeg signals"""

        end_idx = self.start_idx + int(self.duration * self.sfreq)
        time_window = self.times[self.start_idx:end_idx]
        data_window = self.data[:, self.start_idx:end_idx]

        # update xlim
        self.ax_eeg.set_xlim(time_window[0], time_window[-1])

        # update ylim
        self.ax_eeg.set_ylim(-self.dy, self.n_rows * self.dy)
        yticks_loc = np.arange(self.n_rows) * self.dy
        self.ax_eeg.set_yticks(yticks_loc, labels=self.channels)

        # map 'ALL' label to all channels
        all_labels = self.df[self.df['channel'] == 'ALL']

        # update eeg signals
        for i, (ch_name, data_col, line, label_line) in enumerate(
                zip(self.channels, data_window[::-1], self.lines, self.label_lines)):
            line.set_data(time_window, data_col + i * self.dy)

            # obtain current label
            channel_labels = self.df[self.df['channel'] == ch_name]
            relevant_labels = pd.concat([channel_labels, all_labels])


            x_values = []
            y_values = []
            for _, row in relevant_labels.iterrows():
                segment_mask = (time_window >= row['start_time']) & (time_window <= row['stop_time'])
                x_segment = time_window[segment_mask]
                y_segment = (data_col + i * self.dy)[segment_mask]

                if len(x_segment) > 0:
                    if x_values:
                        x_values.append(np.nan)
                        y_values.append(np.nan)
                    x_values.extend(x_segment)
                    y_values.extend(y_segment)

            label_line.set_data(x_values, y_values)

        # update auto marks line
        for line, event_time in zip(self.event_lines, self.events_df["onset"]):
            if time_window[0] <= event_time <= time_window[-1]:
                line.set_xdata([event_time])
                line.set_alpha(1)
            else:
                line.set_alpha(0)

        self.fig.canvas.draw_idle()

    def prev_window(self, event):
        self.start_idx = max(0, self.start_idx - int(20 * self.sfreq))
        self.s_time.set_val(self.start_idx / self.sfreq)
        self.plot_data()

    def next_window(self, event):
        self.start_idx = min(self.max_idx, self.start_idx + int(20 * self.sfreq))
        self.s_time.set_val(self.start_idx / self.sfreq)
        self.plot_data()

    def scale_up(self, event):
        self.scale_factor *= 1.2
        self.dy *= 1.2
        self.plot_data()

    def scale_down(self, event):
        self.scale_factor /= 1.2
        self.dy /= 1.2
        self.plot_data()

    def slider_update(self, val):
        self.start_idx = int(val * self.sfreq)
        self.plot_data()

EEGInteractivePlot(processed_data_path,subject_ID, run_number,
                   label_path)
