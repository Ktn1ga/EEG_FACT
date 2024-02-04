import matplotlib.pyplot as plt
import mne
import numpy as np
import scipy.io as scio
from eegDataset.util_dataset import standardize_data
from scipy.io import loadmat, savemat
import os
import resampy
import csv

import os
import sys
current_path = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(current_path)[0]
sys.path.append(current_path)
sys.path.append(rootPath)
from scipy import signal
from scipy.signal import detrend

from module.transform import filterBank


def _filter(data):
    fs = 250
    f0 = 50
    q = 35
    b, a = signal.iircomb(f0, q, ftype='notch', fs=fs)
    data = signal.filtfilt(b, a, data)
    return data

class Load_SHU():
    def __init__(self):
        self.map = {}
        file_path = os.path.join(current_path,r'../tool/data2run.csv')
        with open(file_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for i,row in enumerate(csv_reader):
                if i == 0: continue
                if len(row) >= 0:
                    # 提取编号和路径
                    iSub = int(row[0])
                    iSes = int(row[1])
                    datasetPath = row[2]
                    self.map['{}-{}'.format(iSub,iSes)]=datasetPath
        print("get map {}".format(len(self.map)))
        self.chans = None

    def get_epochs(self, id, downsampled=None,epoch_window = [0,4]):
        fs = 1000
        offset = 0
        file_to_load = self.map[id]
        print('loading data from {}...'.format(file_to_load))
        # 加载bdf数据
        raw_bdf = mne.io.read_raw_bdf(file_to_load + "\data.bdf", preload=True)
        evt_bdf = mne.read_annotations(file_to_load + "\evt.bdf")
        raw_bdf.set_annotations(evt_bdf)

        # 原始通道
        ch_names = ["Fpz", "Fp1", "Fp2", "AF3", "AF4", "AF7", "AF8", "Fz", "F1", "F2",
                    "F3", "F4", "F5", "F6", "F7", "F8", "FCz", "FC1", "FC2", "FC3",
                    "FC4", "FC5", "FC6", "FT7", "FT8", "Cz", "C1", "C2", "C3", "C4",
                    "C5", "C6", "T7", "T8", "CP1", "CP2", "CP3", "CP4", "CP5", "CP6",
                    "TP7", "TP8", "Pz", "P3", "P4", "P5", "P6", "P7", "P8", "POz",
                    "PO3", "PO4", "PO5", "PO6", "PO7", "PO8", "Oz", "O1", "O2", ]  # "Pz", repference
        # 选择的通道
        ch_names = ["Fz",
                    "FC3", "FC1", "FCz", "FC2", "FC4",
                    "C5","C3", "C1", "Cz", "C2", "C4","C6",
                    "CP3", "CP1","Pz", "CP2",  "CP4",
                    "P3","POz","P4",
                    "Oz"]

        def process(raw):
            # 选择通道
            raw.pick_channels(ch_names=ch_names,ordered=True)

            # 滤波
            raw.filter(l_freq=4, h_freq=100,fir_design="firwin", skip_by_annotation="edge")
            raw.notch_filter(freqs=50)

            # 去眼电
            # ica = mne.preprocessing.ICA(n_components=5, random_state=97, max_iter=200)
            # ica.fit(raw)
            # raw = ica.apply(raw)
            # 平均参考
            # raw.set_eeg_reference(ref_channels='average',projection=True)

            return raw
        raw_bdf = process(raw_bdf)

        events,event_dict = mne.events_from_annotations(raw_bdf)

        trigger_need = ['1', '2', '3']
        event_id = {label: event_dict[label] for label in trigger_need}

        epochs = mne.Epochs(
            raw_bdf,
            events = events,
            event_id = event_id,
            tmin=0,
            tmax=epoch_window[1],
            baseline=None,
            # detrend=1,
            preload=True,
        )

        fs = 250
        epochs.resample(sfreq=fs)

        # 获取实际想象的4s数据
        data = epochs.get_data()[:,:,-1000:]
        # # 获取具体标签的逆映射字典
        reverse_event_id = {v: k for k, v in event_id.items()}
        labels = [int(reverse_event_id[i]) for i in epochs.events[:, -1]] # 获取事件标签

        x = np.array(data)*1e6

        y = np.array(labels)-1

        print(x.shape)
        print(y.shape)

        # plt.plot(abs(x[2,13,:]))
        # plt.show()

        is_fbcnet = False
        if is_fbcnet == True:
            x_all = []
            self.filterBank = filterBank(
                filtBank=[[4, 8], [8, 12], [12, 16], [16, 20], [20, 24], [24, 28], [28, 32], [32, 36], [36, 40]],
                fs=250)
            for i in range(x.shape[0]):
                xi = self.filterBank(x[i, :, :])
                x_all.append(xi)
            x = np.stack(x_all, axis=0)
            x = np.transpose(x,(0, 3, 1, 2))

        # trials * Chan * time
        eeg_data = {'x_data': x,
                    'y_labels': y,
                    'fs': fs}
        return eeg_data


def get_data_shu(subject, mode="NORM", isStandard=True, doShuffle=False):
    load_raw_data = Load_SHU()
    if mode == 'NORM':
        # 训练集使用1,测试集使用3
        Train_session = [1]
        Test_session = [3]
        x_train_l, y_train_l, x_test_l, y_test_l = [], [], [], []
        for session in Train_session:
            id = '{}-{}'.format(subject,session)
            eeg_data = load_raw_data.get_epochs(id=id,downsampled=4)
            x_train, y_train = eeg_data['x_data'], eeg_data['y_labels']
            x_train_l.append(x_train)
            y_train_l.append(y_train)

        for session in Test_session:
            id = '{}-{}'.format(subject,session)
            eeg_data = load_raw_data.get_epochs(id=id,downsampled=4)
            x_test, y_test = eeg_data['x_data'], eeg_data['y_labels']
            x_test_l.append(x_test)
            y_test_l.append(y_test)

        x_train = np.concatenate(x_train_l,axis=0)
        y_train = np.concatenate(y_train_l,axis=0)
        x_test = np.concatenate(x_test_l,axis=0)
        y_test = np.concatenate(y_test_l,axis=0)

        if len(x_train.shape) == 3:
            n_trial, n_channel, n_timepoint = x_train.shape
        elif len(x_train.shape) == 4:
            n_trial,n_bands,n_channel, n_timepoint = x_train.shape

        print("n_trial",n_trial)
        if doShuffle:
            # 打乱
            random_indices = np.random.permutation(n_trial)
            x_train = x_train[random_indices]
            y_train = y_train[random_indices]

        if isStandard:
            x_train, x_test = standardize_data(x_train, x_test, n_channel)
        return x_train, y_train, x_test, y_test

    elif mode == 'INNER':
        choose_session = [1]

        x_all_l, y_all_l= [], []

        for session in choose_session:
            id = '{}-{}'.format(subject,session)
            eeg_data = load_raw_data.get_epochs(id=id,downsampled=4)
            x_train, y_train = eeg_data['x_data'], eeg_data['y_labels']
            x_all_l.append(x_train)
            y_all_l.append(y_train)

        x_all_l = np.concatenate(x_all_l,axis=0)
        y_all_l = np.concatenate(y_all_l,axis=0)

        x_train = x_all_l[:-30,:,:]
        y_train = y_all_l[:-30]
        x_test = x_all_l[-30:,:,:]
        y_test = y_all_l[-30:]

        n_trial, n_channel, n_timepoint = x_train.shape

        if doShuffle:
            # 打乱
            random_indices = np.random.permutation(n_trial)
            x_train = x_train[random_indices]
            y_train = y_train[random_indices]

        if isStandard:
            x_train, x_test = standardize_data(x_train, x_test, n_channel)
        print("train_trial", x_train.shape[0])
        print("test_trial", x_test.shape[0])
        return x_train, y_train, x_test, y_test
    else:
        raise Exception("'{}' mode is not supported yet!".format(mode))