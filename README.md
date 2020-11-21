# CapstoneDesign20

## Deep Learning (Segmentation)
### Flow Chart
![image](https://user-images.githubusercontent.com/37788686/99873946-1f92e780-2c27-11eb-9fc1-0c7366f36dad.png)
![image](https://user-images.githubusercontent.com/37788686/99873948-24f03200-2c27-11eb-875b-b9b5661bddf9.png)

### Results
![image](https://user-images.githubusercontent.com/37788686/99873950-29b4e600-2c27-11eb-8bb6-78c538642414.png)
mIOU=94.6%

- - -

## Post-Image Process

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
###### Copyright 2020. BornToBeDeeplearning All Rights Reserved
