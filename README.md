# DeepLeaningPython

# Theano installation 

```sh
$ pip install theano 
$ conda install mkl=2017
```

# TensorFlow installation 

```sh
https://www.tensorflow.org/install/install_linux
```

# Get Install Directory for tensorflow

```sh
$ python -c 'import os; import inspect; import tensorflow; print(os.path.dirname(inspect.getfile(tensorflow)))'
```
###### Result = /home/amoussou-djangban/anaconda2/lib/python2.7/site-packages/tensorflow

You can check example in this directory to have an idea on how to use TF. 

# Keras installation 

```sh
$ pip install keras
```

# Check keras version

```sh
$ python -c "import keras; print(keras.__version__)"
```
# Configure keras backend

You must edit or create if not exist keras json conf file in your home directory. 

```sh
~/.keras/keras.json
```
By default you have this :

```sh
{
    "epsilon": 1e-07,
    "floatx": "float32",
    "image_data_format": "channels_last",
    "backend": "tensorflow"
}
```

# Check keras backend

```sh
python -c "from keras import backend; print(backend._BACKEND)"
```

# Configure keras backend by using command line

```sh
KERAS_BACKEND=theano python -c "from keras import backend; print(backend._BACKEND)"
```

# General steps to run basic deepl learing model with keras 

1. Definie your model : Create a Sequential model and configured layers 

2. Compile your model : Specify loss function and optimizers and call the complile() function on the model 

3. Fit your model : Train your model on a sample of data by calling fit() function on the model 

4. Make a prediction : Use the model to generate predictions on new data by calling functions such as evaluate() or predict() on the model 









