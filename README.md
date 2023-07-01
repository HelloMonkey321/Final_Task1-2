# Final_Task1-2  
   本文使用了SimCLR（A Simple Framework for Contrastive Learning of Visual Representations）作为自监督学习算法，用于构建图像表示模型。为实现SimCLR模型，本文采用了经典的ResNet-18网络架构作为基础，以便从图像中提取特征。  
本文SimCLR的实现分为以下几个关键步骤：  
（1）数据预处理：在get_aug函数中，定义了数据增强的方式，包括随机裁剪、水平翻转、颜色抖动等，用于增加数据的多样性。  
（2）构建SimCLR模型：SimCLR模型是由一个预训练的骨干网络和一个MLP投影头组成。在SimCLR类中，通过get_backbone函数获取预训练的骨干网络（本文选用ResNet18），将其最后一层全连接层去掉，即“castrate”操作，并添加一个MLP投影头，用于将特征映射到较低维度的特征空间。  
（3）对比学习损失函数：在NTXent类中，实现了SimCLR的对比学习损失函数。该损失函数基于归一化后的特征向量，使用了两个数据增强样本之间的相似度作为目标。目标是让来自同一图像的样本更加接近，而不同图像的样本则更加远离。  
（4）训练过程：在train函数中，首先将输入图像通过数据增强得到两个视角的样本；然后将这两个样本输入SimCLR模型，计算对比学习损失，并通过反向传播更新模型参数。  
（5）模型保存：在训练过程中，会将每个epoch的模型保存到checkpoint文件中，以便后续的评估和使用。  
本实验选用CIFAR-100数据集，训练集图像数量为50000，测试集图像数量为10000，使用的主干网络为ResNet18，模型为SimCLR。具体的实验参数设置如下：  
batchsize：1024  
learningrate：0.001  
优化器：随机梯度下降（SGD）  
iteration：通过代码“forepochintqdm(range(start_epoch,start_epoch+50))”进行设置  
epoch：50  
lossfunction：交叉熵损失函数（CrossEntropyLoss）  
评价指标：（1）训练集准确率（train_acc）：每个epoch结束时的准确率（总样本数/训练集采样数）（2）测试集准确率（test_acc）：每个epoch结束时的准确率（正确分类的样本数/总样本数）  
实验结果对比：  
【期中作业】监督学习（Supervised Learning）中，ResNet-18的准确率为67.27%。  
【期末作业】自监督学习（Self-Supervised Learning）中，ResNet-18的准确率为49.71%。
