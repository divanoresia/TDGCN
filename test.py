"""
 @Author : kangting
 @File : test.py
"""
import torch
import time
from function.correct_function import generate_testdata
from config import cfg
from model.TDGCN import TDGCN

model = {'TDGCN': TDGCN}

device = cfg['device']

batchsize = cfg['batchsize']
DataLen = cfg['DataLen']
ClassNumber = cfg['ClassNumber']
SampPerClass = cfg['SampPerClass']
x_test = generate_testdata(normalize=False,ClassNumber=ClassNumber,SampPerClass=SampPerClass)
testloader = torch.utils.data.DataLoader(x_test, batch_size=batchsize,
                                          shuffle=True)

Gcn_model = model[cfg['model']](cfg).to(device)

Gcn_model.load_state_dict(torch.load('/root/TDGCN_github/experiment/record_10_class/TDGCN_10/ckpts/epoch_10000_train_0.000126.dict',
    weights_only=False))


start = time.time()
correct = 0
total = 0
with torch.no_grad():
    Gcn_model.eval()
    for data in testloader:
        inputs = data[:, 0:DataLen]
        answers = data[:, DataLen]
        labels = (answers.view(batchsize)).long()
        inputs, labels = inputs.to(device), answers.to(device)
        outputs = Gcn_model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    Accuracy = correct / total
    print('The number of test samples is：' + str(total) + '；The correctly estimated sample size is：' + str(correct) + '。')
    print('Accuracy of the network on the %d test samples: %f' % (
        total, Accuracy))
t2 = time.time()

end = time.time()
print('Time of use: {:.5f} s'.format(end - start))



