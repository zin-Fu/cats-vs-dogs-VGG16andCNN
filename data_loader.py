from config import *
from utils import *
import os
from torch.utils.data import DataLoader
from torchvision import transforms

train_mean, train_std = calc_MeanAndStd()  # 用于数据标准化
# train_mean = [0.4875, 0.4544, 0.4164]
# train_std = [0.2521, 0.2453, 0.2481]

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(5),  # 随机旋转图像，旋转角度范围为-5度到+5度之间
        transforms.RandomHorizontalFlip(),  # 随机水平翻转图像
        transforms.RandomResizedCrop(64, scale=(0.96, 1.0), ratio=(0.95, 1.05)),  # 随机裁剪和缩放图像到指定的大小（64x64像素），裁剪比例在0.95到1.05之间，缩放比例在0.96到1.0之间
        transforms.ToTensor(),  # 将图像转换为张量
        transforms.Normalize(train_mean, train_std)  # 对图像进行标准化处理，将每个通道的像素值减去均值（mean）并除以标准差（std），以使得图像数据在每个通道上的均值为0，标准差为1
    ]),
    'valid': transforms.Compose([
        transforms.Resize([64, 64]),
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std)
    ])
}

train_dataset = CatsDogsDataset(img_dir=os.path.join('dogs-vs-cats', 'train'),
                                transform=data_transforms['train'])
valid_dataset = CatsDogsDataset(img_dir=os.path.join('dogs-vs-cats',     'valid'),
                                transform=data_transforms['valid'])
test_dataset = CatsDogsDataset(img_dir=os.path.join('dogs-vs-cats', 'test'),
                               transform=data_transforms['valid'])

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          drop_last=True,  # 如果最后一个批次的样本数量小于batch_size，则将其丢弃
                          shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=False)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False)
