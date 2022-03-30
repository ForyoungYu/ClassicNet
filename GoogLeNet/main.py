import os
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
from GoogLeNet import *
from tqdm import tqdm


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using {} device.".format(device))

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_dataset = datasets.CIFAR10(root="../Datasets/CIFAR10/train",
                                     train=True,
                                     transform=transform_train,
                                     download=True)

    val_dataset = datasets.CIFAR10(root="../Datasets/CIFAR10/test",
                                    train=False,
                                    transform=transform_test,
                                    download=True)
    train_num = len(train_dataset)
    val_num = len(val_dataset)

    batch_size = 32
    nw = min(os.cpu_count(), batch_size if batch_size > 1 else 0)
    print('Using {} dataloader workers every process'.format(nw))

    train_dataloader = Data.DataLoader(dataset=train_dataset,
                                       batch_size=128,
                                       shuffle=True,
                                       num_workers=4)

    val_dataloader = Data.DataLoader(dataset=val_dataset,
                                      batch_size=128,
                                      shuffle=True,
                                      num_workers=4)

    # 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备：", device)

    # 定义网络
    torch.manual_seed(13)
    model = GoogLeNet(10, init_weights=True).to(device)
    writer = SummaryWriter(log_dir="data/log")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_func = nn.CrossEntropyLoss()

    train_steps = len(train_dataloader)
    print_step = 100
    epoch = 30
    best_acc = 0.0
    save_path = 'data/saved_module/GoogLeNet.pth'

    for e in range(epoch):
        # Train
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_dataloader, file=sys.stdout)
        for step, (b_x, b_y) in enumerate(train_bar):
            if torch.cuda.is_available():
                b_x = b_x.to(device)
                b_y = b_y.to(device)
            optimizer.zero_grad()
            output = model(b_x)
            loss = loss_func(output, b_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            train_bar.desc = "Train epoch[{}/{}] loss:{:.3f}".format(
                e + 1, epoch, loss)
            #  train_steps += 1

            #  # 迭代输出日志
            #  if train_steps % print_step == 0:
            #      # 控制台输出
            #      print("Train: {}, Loss: {}".format(train_steps,
            #                                         round(loss.item(), 3)))
            #      # 记录损失
            #      writer.add_scalar("train loss",
            #                        scalar_value=loss.item(),
            #                        global_step=train_steps)
            #
            #      # 绘制参数分布
            #      for name, param in model.named_parameters():
            #          writer.add_histogram(name,
            #                               param.data.cpu().numpy(),
            #                               train_steps)

        # Validate
        model.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_dataloader, file=sys.stdout)
            for val_images, val_labels in val_bar:
                if torch.cuda.is_available():
                    val_images, val_labels = val_images.to(
                        device), val_labels.to(device)
                outputs = model(val_images)
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels).sum().item()

                val_bar.desc = "Valid epoch[{}/{}]".format(e + 1, epoch)

        val_accurate = acc / val_num
        print('[Epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (e + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(model.state_dict(), save_path)

    print("FINISHED TRAINING, THE BEST ACCURACY IS: {}".format(best_acc))
    #  acc = Test(model, test_dataloader, device)
    #  writer.add_scalar("test_acc", acc, total_test_step)
    #  total_test_step += 1
    #
    #  # 保存模型
    #  torch.save(model.state_dict(),
    #             "data/saved_module/resnet_{}.pkl".format(e))


if __name__ == "__main__":
    main()
