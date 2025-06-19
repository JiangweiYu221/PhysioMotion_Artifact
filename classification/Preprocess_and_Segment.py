"""
Project Name: EEG_Preprocess_and_Segment
Author: Aonan He
Contact: ahee0015@student.monash.edu
Last Updated: 2025-05-22
Description:
This script loads preprocessed EDF-format EEG files in BIDS layout,
resamples them to a target frequency, segments the data based on
artifact labels, and saves each window as a pickle file.
"""

import os
import re
import mne
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

# -----------------------------------------------------------
# Configurable parameters
# -----------------------------------------------------------
root_dir      = " "  # Root BIDS directory: e.g., derivatives/preprocessed_BIDS/
label_dir     = " "  # CSV artifact label path
output_dir    = " "  # Output .pkl directory
resample_freq = 125
window_size   = 3 * resample_freq
stride        = window_size // 2

os.makedirs(output_dir, exist_ok=True)

# -----------------------------------------------------------
# Processing: EDF-BIDS based segmentation
# -----------------------------------------------------------
for subject in sorted(os.listdir(root_dir)):
    subject_path = os.path.join(root_dir, subject)
    if not (os.path.isdir(subject_path) and subject.startswith("sub-")):
        continue

    eeg_folder = os.path.join(subject_path, "eeg")
    if not os.path.isdir(eeg_folder):
        continue

    for edf_file in sorted(f for f in os.listdir(eeg_folder) if f.endswith("_eeg.edf")):
        edf_path = os.path.join(eeg_folder, edf_file)

        # extract subject/run info from filename
        match = re.match(r"sub-(\d+)_task-[a-zA-Z]+_run-(\d+)_eeg\.edf$", edf_file)
        if not match:
            print(f"Filename not matched: {edf_file}")
            continue

        subject_id_str, run_number_str = match.groups()
        subject_id_int = int(subject_id_str)
        run_number_int = int(run_number_str)

        # label CSV
        label_file = os.path.join(label_dir, f"sub{subject_id_int}_run{run_number_int:02d}.csv")
        if not os.path.exists(label_file):
            print(f"Label not found: {label_file}, skipping {edf_file}")
            continue

        try:
            print(f"\nLoading {edf_path}")
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            raw.resample(resample_freq, npad="auto")
            data      = raw.get_data()
            ch_names  = raw.ch_names
            n_channels = data.shape[0]

            labels_df = pd.read_csv(label_file).sort_values("start_time")

            segments = []
            current_start = None
            for _, row in labels_df.iterrows():
                st, lbl = row["start_time"], row["label"]
                if lbl == "open_base":
                    continue
                if current_start is None or st != current_start:
                    segments.append([])
                    current_start = st
                segments[-1].append(row)

            eeg_slices = []
            for segment in segments:
                seg_start = int(segment[0]["start_time"] * resample_freq)
                seg_end   = int(segment[0]["stop_time"] * resample_freq)

                current_sample = seg_start
                segment_data = data[:, current_sample:current_sample + window_size]
                first_window = {
                    "eeg_data": segment_data,
                    "labels": np.zeros(n_channels),
                    "start_time": current_sample / resample_freq,
                    "end_time": (current_sample + window_size) / resample_freq,
                    "artifact_types": set(),
                    "affected_channels": []
                }
                for row in segment:
                    ch, st, sp, lbl = row["channel"], row["start_time"], row["stop_time"], row["label"]
                    i0 = int(st * resample_freq)
                    i1 = int(sp * resample_freq)
                    if i1 >= current_sample and i0 <= current_sample + window_size:
                        first_window["artifact_types"].add(lbl)
                        if ch == "ALL":
                            first_window["labels"][:] = 1
                        elif ch in ch_names:
                            first_window["labels"][ch_names.index(ch)] = 1
                first_window["artifact_types"] = list(first_window["artifact_types"])
                first_window["affected_channels"] = list(np.where(first_window["labels"] == 1)[0])
                eeg_slices.append(first_window)

                while current_sample + window_size <= seg_end:
                    current_sample += stride
                    segment_data = data[:, current_sample:current_sample + window_size]
                    win = {
                        "eeg_data": segment_data,
                        "labels": np.zeros(n_channels),
                        "start_time": current_sample / resample_freq,
                        "end_time": (current_sample + window_size) / resample_freq,
                        "artifact_types": set(),
                        "affected_channels": []
                    }
                    for row in segment:
                        ch, st, sp, lbl = row["channel"], row["start_time"], row["stop_time"], row["label"]
                        i0 = int(st * resample_freq)
                        i1 = int(sp * resample_freq)
                        if i1 >= current_sample and i0 <= current_sample + window_size:
                            win["artifact_types"].add(lbl)
                            if ch == "ALL":
                                win["labels"][:] = 1
                            elif ch in ch_names:
                                win["labels"][ch_names.index(ch)] = 1
                    if current_sample + window_size > seg_end:
                        break
                    win["artifact_types"] = list(win["artifact_types"])
                    win["affected_channels"] = list(np.where(win["labels"] == 1)[0])
                    eeg_slices.append(win)

            out_fname = f"sub{subject_id_int}_run{run_number_int:02d}.pkl"
            out_path = os.path.join(output_dir, out_fname)
            with open(out_path, "wb") as f:
                pickle.dump(eeg_slices, f)
            print(f"Saved → {out_path} ({len(eeg_slices)} segments)")

        except Exception as e:
            print(f"[Error] {edf_file}: {e}")

print("\nAll subjects have been processed.")