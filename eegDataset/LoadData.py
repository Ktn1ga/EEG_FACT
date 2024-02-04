import os
import sys
current_path = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(current_path)[0]
sys.path.append(current_path)
sys.path.append(rootPath)

from eegDataset.LoadData2a import get_data_2a
from eegDataset.LoadDataSHU import get_data_shu


def get_data(data_path, subject, data_type='2a', mode="NORM", isStandard=False, doShuffle=False, low_f = None, high_f = None):
    if data_type == '2a':
        return get_data_2a(os.path.join(data_path,'BCIC42a'), subject,
                           mode = mode, isStandard=isStandard,low_f = low_f,high_f = high_f)
    elif data_type == 'shu':
        return get_data_shu(subject, mode=mode, isStandard=isStandard,doShuffle=doShuffle)
    else:
        raise Exception("Not support dataset")