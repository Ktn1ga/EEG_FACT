from models.EEGNet import EEGNet
from models.FBCNet import FBCNet
from models.EEGNet_Inc import EEGNet_Inc
from models.LMDA import LMDA
from models.EEGConformer import Conformer
# from models.FACT import FACT


import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import confusion_matrix,cohen_kappa_score
import os
import sys
current_path = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(current_path)[0]
sys.path.append(current_path)
sys.path.append(rootPath)


# Divide the validation set
# 划分验证集
def cross_validate(x_data, y_label, kfold, data_seed=2023):
    skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=data_seed)
    for split_train_index, split_validation_index in skf.split(x_data, y_label[:,0]):
        # train
        split_train_x = x_data[split_train_index]
        split_train_y = y_label[split_train_index]
        split_train_x, split_train_y = torch.FloatTensor(split_train_x), torch.LongTensor(split_train_y)
        #valid
        split_validation_x = x_data[split_validation_index]
        split_validation_y = y_label[split_validation_index]
        split_validation_x, split_validation_y = torch.FloatTensor(split_validation_x), torch.LongTensor(
            split_validation_y)
        # Index
        split_train_index = torch.LongTensor(split_train_index)
        split_validation_index = torch.LongTensor(split_validation_index)

        # create Dataset
        split_train_dataset = TensorDataset(split_train_x, split_train_y,split_train_index)
        split_validation_dataset = TensorDataset(split_validation_x, split_validation_y,split_validation_index)

        yield split_train_dataset, split_validation_dataset,split_train_index,split_validation_index


# Calculate the kappa coefficient
# 计算kappa系数
def kappa_from_confusion_matrix(confusion_matrix):
    # Total number of instances
    total = np.sum(confusion_matrix)
    # Sum of rows and columns
    sum_rows = np.sum(confusion_matrix, axis=1)
    sum_cols = np.sum(confusion_matrix, axis=0)

    # Expected agreement
    expected = np.sum(sum_rows * sum_cols) / total
    # Observed agreement
    observed = np.trace(confusion_matrix)

    # Kappa calculation
    kappa = (observed - expected) / (total - expected)
    return kappa


# verification model
# 验证模型
def validate_model(model, dataset, device, losser, batch_size=128, n_calsses=4):
    loader = DataLoader(dataset, batch_size=batch_size)
    loss_val = 0.0
    accuracy_val = 0.0
    confusion_val = np.zeros((n_calsses, n_calsses), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for inputs, target, _ in loader:
            inputs = inputs.to(device)
            target = target.to(device)
            if len(target.shape)>1:
                target = target[:, 0]
            probs = model(inputs)
            loss = losser(probs, target)
            loss_val += loss.detach().item()
            accuracy_val += torch.sum(torch.argmax(probs, dim=1) == target, dtype=torch.float32)
            y_true = target.to('cpu').numpy()
            y_pred = probs.argmax(dim=-1).to('cpu').numpy()
            classes = list(range(n_calsses))
            confusion_val += confusion_matrix(y_true, y_pred,labels=classes)
        loss_val = loss_val / len(loader)
        accuracy_val = accuracy_val / len(dataset)
        kappa_score = kappa_from_confusion_matrix(confusion_val)
    return loss_val, accuracy_val, confusion_val,kappa_score


# get model
# 获取模型结构
def getModel(model_name,device,nChan=22,nTime=1000,nClass = 4):
    # Select the model
    if model_name == 'FACT':
        raise Exception("'{}' model is not supported yet!".format(model_name))
        # model = FACT(nChan=nChan, nTime=nTime, nClass=nClass)
    elif model_name == 'EEGNet':
        model = EEGNet(chunk_size=nTime,
                       num_electrodes=nChan,
                       dropout=0.5,
                       kernel_1=64,
                       kernel_2=16,
                       F1=8,
                       F2=16,
                       D=2,
                       num_classes=nClass)
    elif model_name == "FBCNet":
        model = FBCNet(
            num_electrodes=nChan,
            chunk_size=nTime,
            in_channels=9,
            num_classes=nClass,
        )
    elif model_name == 'EEG_Inc':
        model = EEGNet_Inc(channels=nChan, n_classes=nClass)
    elif model_name == 'LMDA':
        model = LMDA(chans=nChan, samples=nTime, num_classes=nClass, depth=9, kernel=75, channel_depth1=24, channel_depth2=9,
             ave_depth = 1, avepool=25)
    elif model_name == "Conformer":
        model = Conformer()
    else:
        raise Exception("'{}' model is not supported yet!".format(model_name))
    model = model.to(device)
    return model