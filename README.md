# Flower Classification with TPUs 

This project evaluates the usability of data augmentation on an imbalanced dataset and provides a console app for predicting flower species from a file. The MobileNetV2 architecture was fine-tuned on this [Kaggle competition dataset](https://www.kaggle.com/c/flower-classification-with-tpus/data) and a part of [Oxford 102 Flower](https://www.kaggle.com/szacho/oxford-102-for-tpu-competition). The first model used [AugMix](https://arxiv.org/pdf/1912.02781.pdf) data augmentation in concert with a special loss function. Both AugMix and special loss were implemented from scratch in TensorFlow and are available in this repository. The second model was trained with no augmentation. 

## Usage
### AugMix
Import and print docstring.
```python
from augmentation.augmix import augmix
print(augmix.__doc__)
```
```
    Performs AugMix data augmentation on given image.

    Parameters:
    image (tensor): an image tensor with shape (x, x, 3) and values scaled to range [0, 1]
    severity (int): level of a strength of transformations (from 1 to 10)
    width (int): number of different chains of transformations to be mixed
    depth (int): number of transformations in one chain, -1 means random from 1 to 3

    Returns:
    tensor: augmented image
```
**Example 1** - transforming single image
```python
from PIL import Image
import numpy as np
import tensorflow as tf
from augmentation.augmix import augmix
# preprocess
image = np.asarray(Image.open('./test_images/sunflower.jpg'))
image = tf.convert_to_tensor(image)
image = tf.cast(image, dtype=tf.float32)
image = tf.image.resize(image, (224, 224)) # resize to square
image /=  255  # scale to [0, 1]
# augment
augmented = augmix(image, severity=6, width=3, depth=-1)
```
**Example 2** - transforming dataset to use with the Jensen-Shannon loss
```python
# here a dataset is a tf.data.Dataset object
dataset = dataset.map(lambda  img, label: (img, augmix(img), augmix(img), label))
```
**Visualization** of AugMix
![visualization of augmix](https://raw.githubusercontent.com/szacho/flower-classification/master/augmentation/augmix_vis.png)
### Predicting from a file
To predict a flower specie from a .jpg file you need:
- jpg image file (like one in /test_images directory)
- h5 trained model (see /models folder)
- labels mapping dictionary (like one in file flower_mapping.json)
- installed: 
	- numpy
	- tensorflow (version 2.x.x)
	- pillow

Then, you will be able to run ```predict.py``` script. It takes 2 positional and 2 optional arguments as described in help:

```>> python predict.py -h```
```
usage: predict.py [-h] [--top_k TOP_K] [--category_names CATEGORY_NAMES]
                  image_path model_path

Simple image classification script.

positional arguments:
  image_path            path to the flower image
  model_path            path to saved model (.h5)

optional arguments:
  -h, --help            show this help message and exit
  --top_k TOP_K         number of top probabilities to display
  --category_names CATEGORY_NAMES
                        dictionary for mapping labels (json file)
```
**Example**

```>> python predict.py test_images/passion_flower.jpg models/mobilenet_adam.h5 --top_k 5```
```
Top 5 probabilities for image test_images/passion_flower.jpg:
0.999962 - passion flower
0.000026 - spear thistle
0.000005 - toad lily
0.000001 - great masterwort
0.000001 - wild rose
```

## More information
- [AugMix](https://arxiv.org/pdf/1912.02781.pdf)
- [Tensor Processing Unit (TPU)](https://www.kaggle.com/docs/tpu) 

## Conclusion
After evaluating models with several configurations, there is clearly no benefit from data augmentation on this dataset. Model with AugMix performs as good as with no augmentation at all, the reason might be that this dataset is strongly imbalanced. 
 
 ## License
This flower classification project is released under MIT License. Do whatever you want. 
