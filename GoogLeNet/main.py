import torch
import torch.nn as nn
from torch.optim import Adam
import torch.utils.data as Data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

from GoogLeNet import GoogLeNet
from functions import *


def main():
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    transform_test = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_dataset = datasets.CIFAR10(root="../Datasets/CIFAR10/train",
                                     train=True,
                                     transform=transform_train,
                                     download=True)

    test_dataset = datasets.CIFAR10(root="../Datasets/CIFAR10/test",
                                    train=False,
                                    transform=transform_test,
                                    download=True)

    train_dataloader = Data.DataLoader(dataset=train_dataset,
                                       batch_size=128,
                                       shuffle=True,
                                       num_workers=4)

    test_dataloader = Data.DataLoader(dataset=test_dataset,
                                      batch_size=128,
                                      shuffle=True,
                                      num_workers=4)

    # 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备：", device)

    # 定义网络
    torch.manual_seed(13)
    model = GoogLeNet(10).to(device)
    writer = SummaryWriter(log_dir="data/log")
    optimizer = Adam(model.parameters(), lr=0.003)
    loss_func = nn.CrossEntropyLoss()

    total_train_step = 0
    total_test_step = 0
    epoch = 80
    print_step = 100
    acc = 0.0

    for e in range(epoch):
        print("================= EPOCH: {}/{}, ACC: {} ===============".format(
            e + 1,epoch, acc))
        # Train
        model.train()
        for step, (b_x, b_y) in enumerate(train_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            output = model(b_x)
            loss = loss_func(output, b_y)
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
                for name, param in model.named_parameters():
                    writer.add_histogram(name,
                                         param.data.cpu().numpy(),
                                         total_train_step)

        # Test
        correct = 0.0
        total = 0
        model.eval()
        with torch.no_grad():
            for data in test_dataloader:
                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                pred = outputs.argmax(dim=1)
                total += inputs.size(0)
                correct += torch.eq(pred, targets).sum().item()
                acc = correct / total
                writer.add_scalar("test_acc", acc, total_test_step)

        total_test_step += 1

        # 保存模型
        torch.save(model.state_dict(),
                   "data/saved_module/googelnet_{}.pkl".format(e))


if __name__ == "__main__":
    main()
