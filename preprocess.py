"""
Project Name:
Author: Jiangwei Yu
Contact: yjiangwei210@163.com
Last Updated: 2025-03-18
Copyright (c) 2025 Jiangwei Yu
Licensed under the MIT License
Project URL:
Description:
This code implements a preprocessing function of the raw bdf data.
all preprocessing steps include data concatenating, 50Hz notch filtering,
0.5-150Hz bandpass filtering, Bipolar Montage and BIDS format conversion.

"""

import mne
import numpy as np
from mne_bids import BIDSPath, write_raw_bids, read_raw_bids

# the root path saving the raw data
root = ""
# the target path saving the processed data
save_dir = ""

for id in range(1, 31):
    subID = str(id)
    for run in range(1,7):
        # raw BIDS path format
        session = 'raw'
        task = 'artifact'
        datatype = 'eeg'
        read_bids_path = BIDSPath(subject=subID, session=session, task=task,
                             run=run, root=root, datatype=datatype)
        raw = read_raw_bids(read_bids_path)
        raw.load_data()

        # 50Hz notch filtering
        freqs = np.arange(50, 251, 50)
        raw = raw.notch_filter(freqs=freqs)

        # 0.5-150Hz bandpass filtering
        raw = raw.filter(h_freq=150, l_freq=0.5)

        # Bipolar Montage
        L_anode = ["Fp1","F7","T7","P7","Fp1","F3","C3","P3","Fz","Cz",
                   "Fp2","F4","C4","P4","Fp2","F8","T8","P8"]
        L_cathode = ["F7","T7","P7","O1","F3","C3","P3","O1","Cz","Pz",
                     "F4","C4","P4","O2","F8","T8","P8","O2"]
        T_anode = ["F7","Fp1","F7","F3","Fz","F4","T7","C3","Cz","C4","P7","P3","Pz","P4","O1","O2"]
        T_cathode = ["Fp1","Fp2","F3","Fz","F4","F8","C3","Cz","C4","T8","P3","Pz","P4","P8","O2","P8"]
        L_pick = ['Fp1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
                  'Fz-Cz', 'Cz-Pz', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fp2-F8', 'F8-T8', 'T8-P8', 'P8-O2']
        T_pick = ['F7-Fp1', 'Fp1-Fp2', 'F7-F3', 'F3-Fz', 'Fz-F4', 'F4-F8', 'T7-C3',
                  'C3-Cz', 'Cz-C4', 'C4-T8', 'P7-P3', 'P3-Pz', 'Pz-P4', 'P4-P8', 'O1-O2', 'O2-P8']
        L_pick.extend(T_pick)
        L_raw_bip_ref = mne.set_bipolar_reference(raw, anode=L_anode, cathode=L_cathode,drop_refs=False)
        LT_raw_bip_ref = mne.set_bipolar_reference(L_raw_bip_ref, anode=T_anode, cathode=T_cathode)
        bip = LT_raw_bip_ref.pick(L_pick)

        # BIDS format conversion
        write_bids_path = BIDSPath(subject=subID, session='preprocessed', run=run,
                             datatype='eeg', root=save_dir, task='artifact')
        write_raw_bids(bip, bids_path=write_bids_path, allow_preload=True, format="BrainVision", overwrite=True)





