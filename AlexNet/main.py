import sys

sys.path.append("..")
from functions import Test
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from AlexNet import AlexNet


def main():
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    #  CIFAR10
    train_dataset = datasets.CIFAR10(root="../Datasets/CIFAR10/train",
                                     train=True,
                                     transform=transform_train,
                                     download=False)
    test_dataset = datasets.CIFAR10(root="../Datasets/CIFAR10/test",
                                    train=False,
                                    transform=transform_test,
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
    model = AlexNet(10).to(device)

    learning_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()

    total_train_step = 0
    total_test_step = 0
    print_step = 100
    epoch = 100
    acc = 0.0

    writer = SummaryWriter(log_dir="data/log")
    for e in range(epoch):
        print("================= EPOCH: {}/{}, ACC: {} ===============".format(
            e + 1, epoch, acc))
        for step, data in enumerate(train_dataloader):
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            output = model(imgs)
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
                for name, param in model.named_parameters():
                    writer.add_histogram(name,
                                         param.data.cpu().numpy(),
                                         total_train_step)

        writer.add_scalar("test_acc", Test(model, test_dataloader, device),
                          total_test_step)
        total_test_step += 1

        # 保存模型
        torch.save(model.state_dict(), "data/saved_module/alexnet_param.pkl")

    writer.close()
    print("训练结束，模型已保存")


if __name__ == "__main__":
    main()
