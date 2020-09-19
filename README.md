# EAST-CRNN
使用EAST训练检测锚点位置，使用CRNN识别文字内容
![](https://upload-images.jianshu.io/upload_images/944794-b2a6f643786f9fdd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
# 安装
```shell
pip install -r requirements.tx
```

训练集：
1. [Google Drive](https://drive.google.com/file/d/1wJWjNvpROtTeTGHHwHugn33OIOeJawVx/view?usp=sharing)
2. [Baidu 网盘链接](https://pan.baidu.com/s/1FX_CRQj8CImAMA1T8rIidA)  密码: gj0j
 
测试集: 
1. [Google Drive](https://drive.google.com/file/d/1Q3R4RHKh8G--0EbioEOl1QUq7lmtJj0U/view?usp=sharing) 
2. [Baidu 网盘链接](https://pan.baidu.com/s/1jYmRh7sSSVpsG070Y_h_ww)  密码: 0kdu

下载训练集解压到当前目录，结构如下面：
```
data
  |___ img
  |___ gt
```
下载权重[vgg16_bn-6c64b313.pth](https://download.pytorch.org/models/vgg16_bn-6c64b313.pth)到pths文件夹下面，如果没有就创建pths文件夹


# 训练

1. 训练east：
```bash
python script/train_east.py
```
2. 训练crnn:
```bash
python script/train_crnn.py
```
# 测试
默认使用sample中的图片进行测试，且测试代码没有进行批量处理，所以速度不会很快，
```bash
python demo.py --crnn pths/crnn_20.pth --east pths/east_50.pth --output output
```
执行完成查看效果

# 其他

east采用advance east，使用vgg16作为主干网络，换成默认的PVAnet应该会加快速度
