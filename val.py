from data_loader import *
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class UnNormalize(object):  # 将经过标准化处理的图像数据恢复到原始的像素值范围
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):  # Args: tensor (Tensor): Tensor image of size (C, H, W) to be normalized. Returns: Tensor: Normalized image.
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)  # The normalize code -> t.sub_(m).div_(s)
        return tensor

def evaluation_and_show(model, test_loader):
    model.eval()
    with torch.set_grad_enabled(False):
        test_acc, test_loss = compute_accuracy_and_loss(model, test_loader, DEVICE)
        print(f'Test accuracy: {test_acc:.2f}')

    unorm = UnNormalize(mean=train_mean, std=train_std)

    test_loader = DataLoader(dataset=train_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=True)

    for features, targets in test_loader:
        break
    _, predictions = model.forward(features[:8].to(DEVICE))  #  将这个批次的特征数据传递给模型进行前向传播，得到预测结果 predictions 取前 8 个样本进行预测。
    predictions = torch.argmax(predictions, dim=1)  # 对预测结果的每个样本，取出概率最高的类别索引，即预测的类别标签

    d = {0: 'cat', 1: 'dog'}

    fig, ax = plt.subplots(1, 8, figsize=(20, 10))  #  创建一个大小为 (20, 10) 的图像画布，并生成 1 行 8 列的子图
    for i in range(8):
        img = unorm(features[i])
        ax[i].imshow(np.transpose(img, (1, 2, 0)))
        ax[i].set_xlabel(d[predictions[i].item()])
    plt.show()

