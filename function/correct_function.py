import torch
import numpy as np
import scipy.io as scio
from config import cfg

def generate_traindata(normalize,reproduce,ClassNumber,SampPerClass):
    TrainData = np.zeros([ClassNumber * SampPerClass, cfg['DataLen'] + 1])
    for q in range(ClassNumber):
        path1 = cfg['train_data_dir'] + str(q + 1) + '.mat'
        tmpdata = scio.loadmat(path1)
        tmpdata2 = tmpdata['kaiji']
        for w in range(SampPerClass):
            TrainData[q * SampPerClass + w][0:cfg['DataLen']] = tmpdata2[w, :]
            TrainData[q * SampPerClass + w][cfg['DataLen']] = q
    TrainData_Lable = TrainData[:, -1]
    TrainData_data = TrainData[:, :-1]
    if normalize:
        max_val = np.max(TrainData_data)
        normalize_TrainData = TrainData_data / max_val
        TrainData = np.hstack((normalize_TrainData, TrainData_Lable[:, np.newaxis]))
        if reproduce:
            TrainData = np.tile(TrainData, [10, 1])
            x_train_copy = torch.from_numpy(TrainData).to(torch.float)
            x_train_no_copy = torch.from_numpy(TrainData).to(torch.float)
        else:
            x_train_copy = torch.from_numpy(TrainData).to(torch.float)
            x_train_no_copy = torch.from_numpy(TrainData).to(torch.float)
    else:
        if reproduce:
            TrainData2 = np.tile(TrainData, [10, 1])
            x_train_copy = torch.from_numpy(TrainData2).to(torch.float)
            x_train_no_copy = torch.from_numpy(TrainData).to(torch.float)
        else:
            x_train_copy = torch.from_numpy(TrainData).to(torch.float)
            x_train_no_copy = torch.from_numpy(TrainData).to(torch.float)
    return x_train_copy,x_train_no_copy

def generate_testdata(normalize,ClassNumber,SampPerClass):
    TestData = np.zeros([ClassNumber * cfg['TestPerClass'], cfg['DataLen'] + 1])
    for q in range(ClassNumber):
        path1 = cfg['test_data_dir'] + str(q + 1) + '.mat'
        tmpdata = scio.loadmat(path1)
        tmpdata2 = tmpdata['kaiji']
        for w in range(cfg['TestPerClass']):
            TestData[q * cfg['TestPerClass'] + w][0:cfg['DataLen']] = tmpdata2[SampPerClass + w, :]
            TestData[q * cfg['TestPerClass'] + w][cfg['DataLen']] = q
    TestData_Lable = TestData[:, -1]
    TestData_data = TestData[:, :-1]
    if normalize:
        max_val = np.max(TestData_data)
        normalize_TestData = TestData_data / max_val
        TestData = np.hstack((normalize_TestData, TestData_Lable[:, np.newaxis]))
        x_test = torch.from_numpy(TestData).to(torch.float)
    else:
        x_test = torch.from_numpy(TestData).to(torch.float)

    return  x_test