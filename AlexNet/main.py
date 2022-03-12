import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

from AlexNet import AlexNet
from functions import *

def main():
    data_transform = transforms.Compose([transforms.ToTensor()])

    #  CIFAR10
    train_dataset = datasets.CIFAR10(root="../Datasets/CIFAR10/train",
                                     train=True,
                                     transform=data_transform,
                                     download=False)
    test_dataset = datasets.CIFAR10(root="../Datasets/CIFAR10/test",
                                    train=False,
                                    transform=data_transform,
                                    download=False)

    train_dataloader = Data.DataLoader(dataset=train_dataset,
                                       batch_size=64,
                                       shuffle=True,
                                       num_workers=4)
    test_dataloader = Data.DataLoader(dataset=test_dataset,
                                      batch_size=64,
                                      shuffle=True,
                                      num_workers=4)

    # 定义训练设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备：", device)

    # 定义网络
    torch.manual_seed(12)
    network = AlexNet(10).to(device)

    learning_rate = 1e-10
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()

    total_train_step = 0
    total_test_step = 0
    print_step = 100
    epoch = 300

    writer = SummaryWriter(log_dir="data/log")
    for e in range(epoch):
        print("================= EPOCH: {} ===============".format(e + 1))
        for step, data in enumerate(train_dataloader):
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            output = network(imgs)
            loss = loss_func(output, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step += 1

            # 迭代输出日志
            if total_train_step % print_step == 0:
                # 控制台输出
                print("Train: {}, Loss: {}".format(total_train_step,
                                                   loss.item()))
                # 记录损失
                writer.add_scalar("train loss",
                                  scalar_value=loss.item(),
                                  global_step=total_train_step)

                # 绘制参数分布
                for name, param in network.named_parameters():
                    writer.add_histogram(name,
                                         param.data.cpu().numpy(),
                                         total_train_step)

                # 计算精度
                with torch.no_grad():
                    for data in test_dataloader:
                        imgs, targets = data
                        imgs = imgs.to(device)
                        targets = targets.to(device)
                        outputs = network(imgs)
                        _, pre_lab = torch.max(outputs, 1)
                        acc = accuracy_score(targets, pre_lab)
                        #  acc_list.append(acc)
                        writer.add_scalar("test_acc", acc, total_test_step)

                total_test_step += 1

        # 保存模型
        torch.save(network.state_dict(), "data/saved_module/alexnet_param.pkl")

    writer.close()
    print("训练结束，模型已保存")


if __name__ == "__main__":
    main()
