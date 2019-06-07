# Chinese-calligraphy-recognition
汉字书法字体识别
共100个汉字，每个字相应400张图片。我们将90%的数据用于网络的训练，10%的数据用于所得模型的验证。

main.py文件中包含数据的导入，预处理，optimizer与criterion的设定，验证集训练集的划分以及模型的训练和训练后模型的验证，并且import了net文件中的网络结构。

3_layer_CNN.py中包含一个最开始为尝试而建立的3层卷积神经网络。


net.py文件中包含两个网络结构：
初始搭建的简单11层CNN（类似VGG-16网络）与Resnet（实际所用模型）。
