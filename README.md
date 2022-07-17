# Learning with Recoverable Forgetting


["Learning with Recoverable Forgetting"](https://arxiv.org/)

Jingwen Ye, Yifang Fu, Jie Song, Xingyi Yang, Songhua Liu, Xin Jin, Mingli Song, Xinchao Wang

In ECCV 2022 


## Overview 

* We propose the concept of **LIRF**. This code doesn't include the network pruning. The full code will be updated later.


## Prerequisite
We use Pytorch 1.4.0, and CUDA 10.1. You can install them with  
~~~
conda install pytorch=1.4.0 torchvision=0.5.0 cudatoolkit=10.1 -c pytorch
~~~   
It should also be applicable to other Pytorch and CUDA versions.  


Then install other packages by
~~~
pip install -r requirements.txt
~~~

## Usage 


### Original networks 

##### Step 1: Train the original network   

~~~
python train_scratch-2.py --save_path [XXX]
~~~



##### Step 2: Train LIRF
~~~
python train_deposit.py --save_path [XXX]
~~~




## Acknowledgement
* [Nasty Teacher](https://github.com/VITA-Group/Nasty-Teacher)

