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

from LeNet import LeNet


def main():
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    # 训练数据
    train_dataset = datasets.MNIST(root="../Datasets/MNIST/train",
                                   train=True,
                                   transform=transform_train,
                                   download=True)

    test_dataset = datasets.MNIST(root="../Datasets/MNIST/test",
                                  train=False,
                                  transform=transform_test,
                                  download=True)

    # 数据加载器
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
    model = LeNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_func = nn.CrossEntropyLoss()

    total_train_step = 0
    total_test_step = 0
    epoch = 100
    print_step = 100
    acc = 0.0

    writer = SummaryWriter(log_dir="data/log")
    for e in range(epoch):
        print("================= EPOCH: {}/{}, ACC: {} ===============".format(
            e + 1, epoch, acc))
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

        writer.add_scalar("test_acc", Test(model, test_dataloader, device),
                          total_test_step)
        total_test_step += 1

        # 保存模型
        torch.save(model.state_dict(),
                   "data/saved_module/lenet_param_{}.pkl".format(e))

    writer.close()
    print("训练结束，模型已保存")


if __name__ == "__main__":
    main()
