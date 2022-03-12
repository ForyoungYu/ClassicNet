import torch.cuda as cuda
import numpy as np


def accuracy_score(lable, pre_lac):
    """ 计算精确度 """
    if cuda.is_available():
        lable = lable.cpu()
        pre_lac = pre_lac.cpu()

    data_len = len(lable)
    output = np.array(lable == pre_lac).sum()
    acc = output / data_len

    return acc


#  A = torch.tensor([1, 2, 3, 4, 5])
#  B = torch.tensor([1, 2, 3, 5, 1])
#  output = accuracy_score(A, B)
#  print(output)
