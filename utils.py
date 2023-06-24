import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

def calc_PicNum():  # calc tot_pic number
    num_train_cats = len([i for i in os.listdir(os.path.join('dogs-vs-cats', 'train'))
                          if i.endswith('.jpg') and i.startswith('cat')])

    num_train_dogs = len([i for i in os.listdir(os.path.join('dogs-vs-cats', 'train'))
                          if i.endswith('.jpg') and i.startswith('dog')])

    print(f'Training set cats: {num_train_cats}')
    print(f'Training set dogs: {num_train_dogs}')

def show_Onepic(picName):  # show
    img = Image.open(os.path.join('dogs-vs-cats', 'train', picName))
    # print(np.asarray(img, dtype=np.uint8).shape)
    plt.imshow(img)
    plt.show()

def move_pic():
    # Move 2500 images from the training folder into a test set folder & 2500 to validation set folder
    if not os.path.exists(os.path.join('dogs-vs-cats', 'test')):
        os.mkdir(os.path.join('dogs-vs-cats', 'test'))
    if not os.path.exists(os.path.join('dogs-vs-cats', 'valid')):
        os.mkdir(os.path.join('dogs-vs-cats', 'valid'))

    for fname in os.listdir(os.path.join('dogs-vs-cats', 'train')):
        if not fname.endswith('.jpg'):
            continue
        _, img_num, _ = fname.split('.')  # 将文件名按'.'进行拆分，获取图像编号
        filepath = os.path.join('dogs-vs-cats', 'train', fname)
        img_num = int(img_num)
        if img_num > 11249:
            os.rename(filepath, filepath.replace('train', 'test'))
        elif img_num > 9999:
            os.rename(filepath, filepath.replace('train', 'valid'))


class CatsDogsDataset(Dataset):

    def __init__(self, img_dir, transform=None):

        self.img_dir = img_dir
        self.img_names = [i for i in os.listdir(img_dir) if i.endswith('.jpg')]  # 获取图像文件夹中以'.jpg'结尾的文件名，并存储在实例变量img_names中

        self.y = []
        for i in self.img_names:
            if i.split('.')[0] == 'cat':
                self.y.append(0)
            else:
                self.y.append(1)

        self.transform = transform

    def __getitem__(self, index):   # 根据索引获取图像和标签
        img = Image.open(os.path.join(self.img_dir, self.img_names[index]))

        if self.transform is not None:
            img = self.transform(img)

        label = self.y[index]
        return img, label

    def __len__(self):
        return len(self.y)

def calc_MeanAndStd():

    custom_transform1 = transforms.Compose([transforms.Resize([64, 64]),
                                           transforms.ToTensor()])
    train_dataset = CatsDogsDataset(img_dir=os.path.join('dogs-vs-cats', 'train'),
                                    transform=custom_transform1)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=5000,
                              shuffle=False)
    train_mean = []
    train_std = []

    for i, image in enumerate(train_loader, 0):  # 将 train_loader 中的每个批次数据与其对应的索引进行配对，并从索引 0 开始递增
        numpy_image = image[0].numpy()

        batch_mean = np.mean(numpy_image, axis=(0, 2, 3))  # 计算批次的均值，对图像的通道维度进行求均值操作
        batch_std = np.std(numpy_image, axis=(0, 2, 3))  # 对图像的通道维度求标准差

        train_mean.append(batch_mean)
        train_std.append(batch_std)

    train_mean = torch.tensor(np.mean(train_mean, axis=0))
    train_std = torch.tensor(np.mean(train_std, axis=0))

    return train_mean, train_std

def compute_accuracy_and_loss(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    cross_entropy = 0.
    for i, (features, targets) in enumerate(data_loader):

        features = features.to(device)
        targets = targets.to(device)

        logits, probas = model(features)
        cross_entropy += F.cross_entropy(logits, targets).item()
        _, predicted_labels = torch.max(probas, 1)  # 找到概率最高的预测类别，并将其索引存储在 predicted_labels 中
        num_examples += targets.size(0)  # 将当前批次样本的数量加到 num_examples 变量中，用于计算准确率
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100, cross_entropy/num_examples




