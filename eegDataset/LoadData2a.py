import mne
import numpy as np
import scipy.io as scio
from eegDataset.util_dataset import standardize_data
import os
import matplotlib.pyplot as plt

class Load_BCIC42a():
    def __init__(self, data_path, persion):
        self.stimcodes_train = {'769', '770', '771', '772'}
        self.stimcodes_test = {'783'}
        self.data_path = data_path
        self.persion = str(persion)
        self.channels_to_remove = ['EOG-left', 'EOG-central', 'EOG-right']

    def get_epochs_train(self, tmin=-0., tmax=2, low_freq=None, high_freq=None, baseline=None, downsampled=None):
        file_to_load = 'A0' + str(self.persion) + 'T.gdf'
        raw_data = mne.io.read_raw_gdf(os.path.join(self.data_path,file_to_load), preload=True)
        if low_freq and high_freq:
            raw_data.filter(l_freq=low_freq, h_freq=high_freq)
        if downsampled is not None:
            raw_data.resample(sfreq=downsampled)
        self.fs = raw_data.info.get('sfreq')
        events, event_ids = mne.events_from_annotations(raw_data)
        stims = [value for key, value in event_ids.items() if key in self.stimcodes_train]
        epochs = mne.Epochs(raw_data, events, event_id=stims, tmin=tmin, tmax=tmax, event_repeated='drop',
                            baseline=baseline, preload=True, proj=False, reject_by_annotation=False)
        epochs = epochs.drop_channels(self.channels_to_remove)
        self.y_labels = epochs.events[:, -1] - min(epochs.events[:, -1])
        self.x_data = epochs.get_data() * 1e6
        # epochs.plot()
        # plt.show()
        # trial channel time
        eeg_data = {'x_data': self.x_data[:, :, :-1],
                    'y_labels': self.y_labels,
                    'fs': self.fs}
        return eeg_data

    def get_epochs_test(self, tmin=-0., tmax=2, low_freq=None, high_freq=None, baseline=None, downsampled=None):
        file_to_load = 'A0' + str(self.persion) + 'E.gdf'
        raw_data = mne.io.read_raw_gdf(os.path.join(self.data_path ,file_to_load), preload=True)
        data_path_label = os.path.join(self.data_path,"true_labels/A0" + self.persion + "E.mat")
        mat_label = scio.loadmat(data_path_label)
        mat_label = mat_label['classlabel'][:, 0] - 1
        if low_freq and high_freq:
            raw_data.filter(l_freq=low_freq, h_freq=high_freq)
        if downsampled is not None:
            raw_data.resample(sfreq=downsampled)
        self.fs = raw_data.info.get('sfreq')
        # raw_data.plot()
        # plt.show()
        events, event_ids = mne.events_from_annotations(raw_data)
        stims = [value for key, value in event_ids.items() if key in self.stimcodes_test]
        epochs = mne.Epochs(raw_data, events, event_id=stims, tmin=tmin, tmax=tmax, event_repeated='drop',
                            baseline=baseline, preload=True, proj=False, reject_by_annotation=False)
        epochs = epochs.drop_channels(self.channels_to_remove)
        self.y_labels = epochs.events[:, -1] - min(epochs.events[:, -1]) + mat_label
        self.x_data = epochs.get_data() * 1e6

        # plt.plot(self.x_data[2,3,:])
        # plt.show()

        eeg_data = {'x_data': self.x_data[:, :, :-1],
                    'y_labels': self.y_labels,
                    'fs': self.fs}
        return eeg_data


def get_data_2a(data_path, subject, mode="NORM", isStandard=True,low_f = None,high_f = None):
    if mode == 'NORM':
        load_raw_data = Load_BCIC42a(data_path, subject)
        eeg_data = load_raw_data.get_epochs_train(tmin=0., tmax=4.,low_freq=low_f,high_freq=high_f)
        x_train, y_train = eeg_data['x_data'], eeg_data['y_labels']
        eeg_data = load_raw_data.get_epochs_test(tmin=0., tmax=4.)
        x_test, y_test = eeg_data['x_data'], eeg_data['y_labels']
        n_trial, n_channel, n_timepoint = x_train.shape
        if isStandard:
            x_train, x_test = standardize_data(x_train, x_test, n_channel)
        return x_train, y_train, x_test, y_test
    elif mode == 'ALL':
        x_train_l, y_train_l, x_test_l, y_test_l = [], [], [], []
        for sub in range(1, 10):
            # single subjeet
            load_raw_data = Load_BCIC42a(data_path, sub)
            eeg_data = load_raw_data.get_epochs_train(tmin=0., tmax=4.)
            x_train, y_train = eeg_data['x_data'], eeg_data['y_labels']
            eeg_data = load_raw_data.get_epochs_test(tmin=0., tmax=4.)
            x_test, y_test = eeg_data['x_data'], eeg_data['y_labels']
            x_train_l.append(x_train)
            y_train_l.append(y_train)
            x_test_l.append(x_test)
            y_test_l.append(y_test)
        x_train = np.concatenate(x_train_l,axis=0)
        y_train = np.concatenate(y_train_l,axis=0)
        x_test = np.concatenate(x_test_l,axis=0)
        y_test = np.concatenate(y_test_l,axis=0)

        n_trial, n_channel, n_timepoint = x_train.shape
        print(n_trial,n_channel,n_timepoint)

        # 打乱
        random_indices = np.random.permutation(n_trial)
        x_train = x_train[random_indices]
        y_train = y_train[random_indices]

        if isStandard:
            x_train, x_test = standardize_data(x_train, x_test, n_channel)
        return x_train, y_train, x_test, y_test
    else:
        raise Exception("'{}' mode is not supported yet!".format(mode))
