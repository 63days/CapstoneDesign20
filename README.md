# CapstoneDesign20(딥러닝을 이용한 2차원 X-선 뼈 영역 분할 및 3차원 뼈 모델링 기법)
Semantic Segmentation of femoral bone using deep learning is performed, and fracture area is shown in 3D by matching with the 3D model. Traditional methods of 
fracture surgery required doctors to use real-time X-ray equipment to understand patients's fracture. But this project will help them understand their fracture situation with just two X-ray. This significantly reduces the time of exposure to radiation when doctors x-ray.

For more information, visit our youtube https://youtu.be/ZWfRwMb3aCY
## Deep Learning (Segmentation)
### Flow Chart
![image](https://user-images.githubusercontent.com/37788686/99873946-1f92e780-2c27-11eb-9fc1-0c7366f36dad.png)
![image](https://user-images.githubusercontent.com/37788686/99873948-24f03200-2c27-11eb-875b-b9b5661bddf9.png)
### Characteristics
We used U-Net, commonly used in biomedical Semantic Segmentation, as a baseline model. But we slightly modified the U-Net model, so we can make a more efficient and high-performance model.

![image](https://user-images.githubusercontent.com/37788686/100350356-1a6dd800-302d-11eb-822f-fc7186275079.png)

The image above is a picture of an existing U-Net architecture. We changed maxpool(2x2) layers to Conv(stride=2) layers and up-conv(2x2) to Transposed Conv layers. So we were able to construct the optimal layer where mathematical operations were performed differently depending on the input value.
```python
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, separable, down_method='maxpool'):
        super().__init__()
        if down_method == 'maxpool':
            self.downsample = nn.MaxPool2d(2)
        elif down_method == 'conv':
            self.downsample = nn.Conv2d(in_channels, in_channels, 2, stride=2, bias=False)

        self.convs = DoubleConv(in_channels, out_channels, separable=separable)

    def forward(self, x):
        x = self.downsample(x)
        x = self.convs(x)
        return x



class Up(nn.Module):
    def __init__(self, in_channels, out_channels, separable, up_method='bilinear'):
        super().__init__()
        self.convs = DoubleConv(in_channels, out_channels, separable=separable)
        if up_method == 'bilinear':
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                PointWiseConv(in_channels, out_channels, bias=False)
            )
        elif up_method == 'transpose':
            self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2,
                                               bias=False)


    def forward(self, bottom_x, skip_x):
        bottom_x = self.upsample(bottom_x) #[B, C, H, W]
        concate_x = torch.cat([skip_x, bottom_x], dim=1)

        return self.convs(concate_x)
```
Down and Up class are the base modules of U-Net. They take down,up_method parameter and modules are constructed accordingly. Down class can consist of convolution(stride=2) layer and Up class of transposed convolution layer.

        
![image](https://user-images.githubusercontent.com/37788686/100351312-9b799f00-302e-11eb-8514-8f7977ff3fbb.png)
```python
class DepthWiseConv(nn.Module):
    def __init__(self, channels, kernel_size, stride=1, padding=0, padding_mode='zeros',
                 dilation=1, bias=True):
        '''
        In Depth-Wise Conv, in_channels and out_channels are same.
        channel-wise convoution.
        '''
        super().__init__()
        self.depthwise_conv = nn.Conv2d(channels, channels, kernel_size, stride,
                                        padding=padding, dilation=dilation,
                                        groups=channels, bias=bias, padding_mode=padding_mode)

    def forward(self, x):
        return self.depthwise_conv(x)

class PointWiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(PointWiseConv, self).__init__()
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.pointwise_conv(x)

class DepthWiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, padding_mode='zeros',
                 dilation=1, bias=True):
        super(DepthWiseSeparableConv, self).__init__()
        self.depthwise_conv = DepthWiseConv(in_channels, kernel_size, stride, padding,
                                            padding_mode, dilation, bias)
        self.pointwise_conv = PointWiseConv(in_channels, out_channels)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x
 ``` 
And we changed all Standard Conv layers to Depthwise Separable Conv layers, so we were able to implement a more efficient model that would reduce the parameter by one-fifth and maintain performance. 
| | mIOU | Params(M) |
| - | --- | -------- |
| Existing U-Net | 94.5% | 31.0 |
| Modified U-Net | 94.6% | 7.3 |

To reduce noise from the labeling image obtained by the deep learning model and obtain a more dense labeling image, we used Efficient Full-Connected CRF.
![image](https://user-images.githubusercontent.com/37788686/100351770-54d87480-302f-11eb-8ec6-a4a594de57c8.png)
![image](https://user-images.githubusercontent.com/37788686/100352779-df6da380-3030-11eb-828b-3b984c66272f.png)

Fully-Connected CRF consists of two kernels, appearance kernel and smoothness kernel. The appearance kernel is inspired by the observation that nearby pixels with similar color are likely to be in the same class. The smoothness kernel removes small isolated regions.

### Results
![image](https://user-images.githubusercontent.com/37788686/99873950-29b4e600-2c27-11eb-8bb6-78c538642414.png)
mIOU=94.6%

- - -

## Post-Image Process
### Similarity
Comparing similarity to determine the angle rotated around the z-axis. Pre-processed in 3 and use the straight line image with the z-axis.

Z-axis rotated standard images are already saved in directory.

To do this we use the mahotas technology algorithm.

This library is used in image processing, which has a simple algorithm and is faster than other matching algorithms.

The principle is

1. Pattern the objects and images to be found in the image using an array.

2. Using the convolution, find the section where the value between the object and the image is the maximum.

#### 1)image_simularity
Using the mahotas library, first resize the bones measured for similarity and the bone to be measured to a size suitable for measurement.

Each picture is patterned by extracting features. At this time, the GLCM algorithm is used,

It is a method of calculating how often a pair of pixels having a specific value occurs in a specific spatial relationship.

Lastly, convolution measures the degree of similarity using the cosine similarity method.

#### 2)plotimage
| name|Input|Output|
|:---:|:---:|:---:|
| Plot_Image|bone_image(not fracture)|similar bone image until second|
| Plot_Image2|bone_image(fracture)|Upper similar bone image until second,Under similar bone image until second|

This function is used to list pictures similar to the input picture. 
In the case of fracture, since pictures should be listed for each case of the upper and lower versions, there are two versions of the function.
### Example of UnderBoneCase
![Under_Bone](https://user-images.githubusercontent.com/53164312/99875415-e82a3800-2c32-11eb-8d29-1b02fa5d4092.png)


* * *
### Internal Function
#### 1) expand
When rotating the image, make it 1.5 times the pixel size starting from the center and fill the gap with black to prevent image loss (cut) at both ends.

|Input|Output|
|:---:|:---:|
|OpenCV2 image|OpenCV2 image(expanded image)|

#### 2) angle
Evaluate the angle at which the bone is rotated around the y-axis in the image.

|Input|Output|
|:---:|:---:|
|OpenCV2 image|integer(angle)|

#### 3) rotate
Rotate the image as expanded by the input angle.

|Input|Output|
|:---:|:---:|
|OpenCV2 image, integer(angle)|OpenCV2 image(rotated image)|

#### 4) division
Divide the area into two parts, and remove any unwanted parts from deep learning.

|Input|Output|
|:---:|:---:|
|OpenCV2 image|OpenCV2 image(divided & removed unwanted part image)|


#### 5) fracWhere
A function for finding features in an image in which a fractured bone overlaps, and for pre-processing to separate a fractured bone from feature points into two images.

|Input|Output|
|:---:|:---:|
|OpenCV2 image|pts1, pts0 (coordinate of fractured position)|

#### 6) seperate
Function of splitting unoverlapped bones

|Input|Output|
|:---:|:---:|
|OpenCV2 image|OpenCV2 image0, OpenCV2 image1|

#### 7) cuttinge
Split overlapping fractured bones using input coordinates.

|Input|Output|
|:---:|:---:|
|OpenCV2 image, pts0, pts1(coordinate of fractured position)|OpenCV2 image0, OpenCV2 image1|


### functions.py
Files Combining Completed Modules

| Internal Function name | functions |
|---|---|
| **expand** | expanding 5 times the image |
| **angle** | Evaluate the rotated angle relative to the y-axis. |
| **rotate** | Rotate the image so that it stands on the z-axis. |
| **division** | Divide and erase unnecessary areas. |
| **seperate** | Divide the fractured bone into two images so that it has only one bone. |
| **fracWhere** | When a fractured bone is  it first finds the feature points and returns them to find the fractured position. |
| **cutting** | Using the coordinates of the feature point, separate the fractured bone into two images so that it has only one bone per image. |

### main.py
| Case Number | Cases |
|---|---|
| **Case 1** | _No_ fracture, _no_ area division required. |
| **Case 2** | Fractured, overlapped_ |
| **Case 3** | Fractured, _not_ overlapped |
| **Case 4** | No_ fracture; area division required. |


- - -

## 3D Model Viewer
Rotating 3D model(bone, * .STL).

Input : angle (x, y, z)


### 1) Load STL
<img src="https://user-images.githubusercontent.com/58382336/98698584-8ad3f280-23b9-11eb-9055-3bfbb126cde9.png"  width="700" height="382">

### 2) Rotation
<img src="https://user-images.githubusercontent.com/58382336/98698681-aa6b1b00-23b9-11eb-9547-a6f6d66ea951.png"  width="700" height="382">

* * *

## Prerequisites
- python==3.7
- torch==1.5.1
- torchvision==0.6.1
- pydensecrf==1.0rc2
- mahotas==1.4.11
- scikit-learn==0.23.1
- opencv-python==4.2.0
- numpy==1.18.1

## References
[1] [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

[2] [DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs](https://arxiv.org/abs/1606.00915)

[3] [Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials](https://arxiv.org/abs/1210.5644)

###### Copyright 2020. BornToBeDeeplearning All Rights Reserved
