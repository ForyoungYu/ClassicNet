import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as Data
from Net import LeNet

test_dataset = datasets.MNIST(root="../Datasets/MNIST/test",
                              train=False,
                              transform=transforms.ToTensor(),
                              download=True)

test_dataloader = Data.DataLoader(dataset=test_dataset,
                                  batch_size=100,
                                  shuffle=True,
                                  num_workers=1)

network = LeNet()
param = torch.load("data/saved_module/lenet_param_22.pkl") # 加载模型
network.load_state_dict(param) # 将参数放入模型当中
acc = []
for data in test_dataloader:
    imgs, targets = data
    output = network(imgs) # 输出预测
    _, pre_lab = torch.max(output, 1) # 提取预测序列
    batch_acc = np.array(pre_lab == targets).sum() / test_dataloader.batch_size
    acc.append(batch_acc)

acc = sum(acc) / len(acc)
print("accuracy: ", acc) # 输出正确率
