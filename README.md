# cats-vs-dogs-VGG16andCNN

#### 这个项目是一个用于训练猫狗图像分类模型的代码库。它包括了构建不同卷积神经网络模型（VGG16和CNN）、加载数据集、训练模型和评估性能的功能

#### 注意，您需要从kaggle官网下载数据集，解压并且将它放到文件目录下，当然您也可以创建自己的数据集

### 文件功能
#### main.py：

导入所需的文件和模块。
计算数据集大小。
显示一张图像。
将训练文件夹中的2500张图像移动到测试集和验证集文件夹。
构建模型。
定义优化器。
在设备上训练模型。
评估模型。
#### model.py：

定义了两个模型类：VGG16和CNN。
VGG16模型是基于VGG16架构的卷积神经网络模型。
CNN模型是一个简单的卷积神经网络模型。
#### train.py：

定义了训练函数train，用于训练模型。
在每个epoch中，进行模型训练和验证。
输出训练和验证的准确率和损失。
可视化训练过程中的损失和准确率曲线。
#### utils.py：

包含一些辅助函数和数据集类。
calc_PicNum函数用于计算训练集中猫和狗的图像数量。
show_Onepic函数用于显示一张图像。
move_pic函数用于将训练集中的图像移动到测试集和验证集。
CatsDogsDataset类是一个自定义的数据集类，用于加载猫狗图像数据集。
calc_MeanAndStd函数用于计算图像数据集的均值和标准差。
#### config.py:

定义了一些模型和训练的超参数配置,可以根据需要修改
#### dataloader.py:
定义了数据加载器和实现数据预处理，用于加载和组织训练集、验证集和测试集的数据
#### val.py:
对训练好的模型进行评估和可视化
### 运行方式
首先您先在config.py去选择调整合适的参数，然后在main文件中选择使用的模型，默认使用的是CNN。

然后在终端运行 python main.py
