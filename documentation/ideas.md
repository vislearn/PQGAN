## Datasets
- https://image-net.org/challenges/LSVRC/2012/2012-downloads.php
- found that the standard ImageNet for comparisons is the ILSVRC2012 from the official imageNet website
- val and test images are in val and test folders in subfolder 0 (=label 0), since this was the easiest way to extract them and to load them in the data loader
- the train dataset is in the train folder and the images are in their respective subfolder
- I found that torch can also transform size on the fly so i kept the images as they are

- dowloaded FFQH from https://www.kaggle.com/datasets/gibi13/flickr-faces-hq-dataset-ffhq