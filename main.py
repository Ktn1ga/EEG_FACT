import os
import sys
import pandas as pd
import datetime
import torch
from torch import nn
import numpy as np
import time
from loguru import logger
from eegDataset.LoadData import get_data
from common.func import getModel
from common.exp import exp_cross_session

current_path = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(current_path)[0]
sys.path.append(current_path)
sys.path.append(rootPath)

# DATA path set
# data_path = "DATA"
data_path = "F:\EEGPT\EEG_Data\data2"

model_name = 'EEGNet'
# FACT, EEGNet, FBCNet, EEG_Inc, LMDA, Conformer

datatype = "2a"
datamode = 'NORM'
isStandard = False
# 2a shu

if datatype == '2a':
    nChan = 22
    nTime = 1000
    nClass = 4
elif datatype == 'shu':
    nChan = 22
    nTime = 1000
    nClass = 3

first_epoch = 3000
early_stop_epoch = 500
second_epoch = 800
batch_size = 32
kfolds = 5
if datatype == 'shu':
    kfolds = 6

np.random.seed(2023)
torch.manual_seed(2023)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = getModel(model_name, device, nChan=nChan, nTime=nTime, nClass=nClass)
losser = nn.CrossEntropyLoss().to(device)


print("HI")
print("current device:" + torch.cuda.get_device_name(device))
print('Trainable Parameters in the network are: ' + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

def main_exp_cross_session():
    if datatype == 'shu':
        sub_num = 67
    elif datatype == '2a':
        sub_num = 9
    res_list = []

    for subject in range(1, sub_num + 1):
        save_path = os.path.join(current_path, 'LOG', 'exp_1215', datatype,
                                 model_name, 's{:}/'.format(subject))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        date = time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))

        # LOG Config
        # 日志设置
        logger.remove()
        logger.add(sink=os.path.join(save_path, 'log{}.log'.format(date)), level="INFO", retention='1 year')
        logger.add(sys.stderr)
        logger.info("SUBJECT:{}".format(subject))
        logger.info(torch.__version__)
        logger.info("current device:" + torch.cuda.get_device_name(device))
        logger.info(model)
        logger.info('Trainable Parameters in the network are: ' + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

        # get data
        # 获取数据
        x_train, y_train, x_test, y_test = get_data(data_path,
                                                    subject=subject,
                                                    mode=datamode,
                                                    data_type=datatype,
                                                    isStandard=isStandard,
                                                    doShuffle=False)
        logger.info("Train_size:", x_train.shape, "Test_size:", x_test.shape)

        # start train
        # 开始训练
        start = datetime.datetime.now()
        res = exp_cross_session(subject=subject, losser=losser, save_path=save_path, model_name=model_name,
                                frist_epoch=first_epoch, eary_stop_epoch=early_stop_epoch,
                                second_epoch=second_epoch,
                                kfolds=kfolds, batch_size=batch_size, device=device,
                                x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                                nChan=nChan, nTime=nTime, nClass=nClass)
        end = datetime.datetime.now()
        res_list.append(res)
        logger.info(f"Subject{subject}  Train Time：{end - start} \n")

    # 记录准确率结果
    df = pd.DataFrame(data=res_list)
    df.to_csv(os.path.join(save_path, "results.csv"), mode='w', header=None, index=None)


if __name__ == "__main__":
    main_exp_cross_session()
