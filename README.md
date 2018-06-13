# VariationalAutoEncoder
A vanilla implementation of Auto-Encoding Variational Bayes using numpy and Python 3 - https://arxiv.org/abs/1312.6114

## To read the MNIST data 

```python
from nn import *
from vae import *

# MNIST digits 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist_data")
plt.imshow(mnist.train.images[0].reshape((28, 28)), interpolation='none', cmap=plt.get_cmap('gray'));

# Frey face data 
import scipy.io as sio
frey_face = sio.loadmat("./files/frey_rawface.mat")
faces = frey_face['ff'].T

plt.imshow(faces[0].reshape((28, 20)), interpolation='none', cmap=plt.get_cmap('gray'));
```

There are two options to learn Frey faces (Gaussian VAE and Bernoulli VAE), MNIST should be learnt using Bernoulli VAE. 

Example of training:

```python
PARAMS = {
  "num_input": 560,
  "num_hidden": 200,
  "num_latent": 2,
  "dropout": .9,
  "hidden_activ": "tanh",    # options: identity, tanh, sigmoid, exp, softmax, elu (will be added soon)
  "output_activ": "sigmoid"
}

vae = BerVAE(PARAMS)
vae.train(X=X_train, batch_size=128, num_iter=20000, step_size=0.01, print_every=200)
```

### Result 

![alt text](https://https://github.com/anhtu/VariationalAutoEncoder/files/mnist_digits_2d.png "MNIST digits in 2D")

![alt text](https://https://github.com/anhtu/VariationalAutoEncoder/files/frey_faces_2d.png "Freyface in 2D")

