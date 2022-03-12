import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import torch.utils.data as Data
from Net import AlexNet

test_data = CIFAR10("../Datasets/CIFAR10/test",
                    train=False,
                    transform=transforms.ToTensor(),
                    download=False)

test_dataloader = Data.DataLoader(dataset=test_data,
                                  batch_size=100,
                                  shuffle=True,
                                  num_workers=4)

network = AlexNet(10)
model = torch.load("data/save_module/alexnet_param.pkl") # 加载模型
network.load_state_dict(model) # 将参数放入模型当中

acc_list = []
for data in test_dataloader:
    imgs, targets = data
    output = network(imgs) # 输出预测
    _, pre_lab = torch.max(output, 1) # 提取预测序列
    acc = np.array(pre_lab == targets).sum() / 100 # 计算正确率

    print("accuracy: ", acc) # 输出正确率
    acc_list.append(acc)

sum = 0
for a in acc_list:
    sum += a

acc = sum / 100
print(acc)
