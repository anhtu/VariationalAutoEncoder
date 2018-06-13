import numpy as np
from nn import *

class BerVAE:
  """Bernoulli VAE used for binary data"""

  def __init__(self, params):
    self.__dict__.update(**params)
    self.num_output = self.num_input
    num_input, num_hidden, dropout, num_latent, num_output = self.num_input, self.num_hidden, self.dropout, self.num_latent, self.num_output

    # encoder
    self.en_hidden = LayerFactory.build(self.hidden_activ, (num_input, num_hidden, dropout))  # W1, b1
    self.en_latent_mu = LayerFactory.build("identity", (num_hidden, num_latent, dropout))     # W2, b2
    self.en_latent_sigma = LayerFactory.build("exp", (num_hidden, num_latent, 1.))            # W3, b3

    # decoder
    self.de_hidden = LayerFactory.build(self.hidden_activ, (num_latent, num_hidden, dropout)) # W4, b4
    self.de_out_mu = LayerFactory.build(self.output_activ, (num_hidden, num_output, 1.))      # W5, b5


  def __call__(self, X_batch, epsilon=None):
    """
    Perform the forward pass

    Inputs:
    - X_batch: A numpy array of shape (num_input, mini_batch) represents the batch training data
    - dropout: (boolean) Whether to activate dropout during forward pass
    - epsilon: A numpy array of shape (num_latent, 1) represents a sample from Standard Normal distribution

    Returns:
    - loss: (scalar) Value of loss function
    """
    num_latent = self.num_latent
    eps = 1e-12
    # encoder
    h_en     = self.en_hidden(X_batch)
    z_mu     = self.en_latent_mu(h_en)
    z_sigma2 = self.en_latent_sigma(h_en)

    # we sample from the posterior q(z|x)
    if epsilon is None: epsilon = np.random.normal(size=num_latent).reshape(num_latent, 1)
    z_samples = z_mu + np.sqrt(z_sigma2) * epsilon

    # decoder
    h_de     = self.de_hidden(z_samples)
    x_mu     = self.de_out_mu(h_de)

    # use the z_samples to estimate the ELBO
    neg_log_bernoulli = lambda x, p: -x*np.log(p + eps) - (1. - x)*np.log(1. - p + eps)
    neg_kl_divergence = 0.5*(-1. - np.log(z_sigma2) + z_mu**2 + z_sigma2)

    # minimize this loss instead of maximizing
    loss = neg_kl_divergence.sum() + neg_log_bernoulli(X_batch, x_mu).sum()

    self.h_en, self.z_mu, self.z_sigma2, self.epsilon, self.z_samples = h_en, z_mu, z_sigma2, epsilon, z_samples
    self.h_de, self.x_mu, self.X_batch = h_de, x_mu, X_batch
    return loss


  def generate_data(self, samples=None):
    # sample from the prior
    z_samples = np.random.normal(size=self.num_latent).reshape(self.num_latent, 1) if samples is None else samples

    # forward to the output
    h_de     = self.de_hidden.predict(z_samples)
    x_mu     = self.de_out_mu.predict(h_de)
    return x_mu


  def back_prop(self):
    eps = 1e-12
    h_en, z_mu, z_sigma2, epsilon, z_samples = self.h_en, self.z_mu, self.z_sigma2, self.epsilon, self.z_samples
    h_de, x_mu, X_batch = self.h_de, self.x_mu, self.X_batch
    dloss_dxm = -X_batch/(x_mu + eps) + (1. - X_batch)/(1. - x_mu + eps)
    dloss_dhde, dloss_dW5, dloss_db5 = self.de_out_mu.back_prop(dloss_dxm)
    dloss_dz_samples, dloss_dW4, dloss_db4 = self.de_hidden.back_prop(dloss_dhde)

    # loss to zm, zs2
    dloss_dzm1  = z_mu
    dloss_dzs1  = -0.5/z_sigma2 + 0.5

    # propagate from z_samples
    dz_samples_dzm2 = 1.
    dz_samples_dzs2 = 0.5*epsilon * (z_sigma2)**-0.5
    dloss_dzm2 = dloss_dz_samples * dz_samples_dzm2
    dloss_dzs2 = dloss_dz_samples * dz_samples_dzs2

    dloss_dzm = dloss_dzm1 + dloss_dzm2
    dloss_dzs = dloss_dzs1 + dloss_dzs2
    dloss_dhen1, dloss_dW2, dloss_db2 = self.en_latent_mu.back_prop(dloss_dzm)
    dloss_dhen2, dloss_dW3, dloss_db3 = self.en_latent_sigma.back_prop(dloss_dzs)
    dloss_dhen = dloss_dhen1 + dloss_dhen2
    _, dloss_dW1, dloss_db1 = self.en_hidden.back_prop(dloss_dhen)

    return dloss_dW5, dloss_db5, dloss_dW4, dloss_db4, dloss_dW3, dloss_db3, dloss_dW2, dloss_db2, dloss_dW1, dloss_db1


  def train(self, X, batch_size=100, num_iter=1000, step_size=0.001, print_every=100):
    """
    Perform training procedure using Adagrad
    """
    W5, b5, W4, b4 = self.de_out_mu.W, self.de_out_mu.b, self.de_hidden.W, self.de_hidden.b
    W3, b3, W2, b2, W1, b1 = self.en_latent_sigma.W, self.en_latent_sigma.b, self.en_latent_mu.W, self.en_latent_mu.b, self.en_hidden.W, self.en_hidden.b
    eps = 1e-12
    num_train = X.shape[1]
    cache = {"W5": 0., "W4": 0., "W3": 0., "W2": 0., "W1": 0., "b5": 0., "b4": 0., "b3": 0., "b2": 0., "b1": 0.}

    for i in range(num_iter+1):
      # create mini-batch
      ix_batch = np.random.choice(range(num_train), size=batch_size, replace=False)
      X_batch  = X[:, ix_batch]

      loss = self.__call__(X_batch)
      dW5, db5, dW4, db4, dW3, db3, dW2, db2, dW1, db1 = self.back_prop()

      # Adagrad update
      cache["W5"] += dW5**2
      cache["W4"] += dW4**2
      cache["W3"] += dW3**2
      cache["W2"] += dW2**2
      cache["W1"] += dW1**2
      cache["b5"] += db5**2
      cache["b4"] += db4**2
      cache["b3"] += db3**2
      cache["b2"] += db2**2
      cache["b1"] += db1**2

      def ratio_weight_update(dW, W, W_name):
        param_scale = np.linalg.norm(W.ravel())
        update = step_size * dW / (np.sqrt(cache[W_name]) + eps)
        update_scale = np.linalg.norm(update.ravel())
        return update_scale / param_scale
      ratio_W1 = ratio_weight_update(dW1, W1, "W1")

      W5 -= step_size * dW5 / (np.sqrt(cache["W5"]) + eps)
      W4 -= step_size * dW4 / (np.sqrt(cache["W4"]) + eps)
      W3 -= step_size * dW3 / (np.sqrt(cache["W3"]) + eps)
      W2 -= step_size * dW2 / (np.sqrt(cache["W2"]) + eps)
      W1 -= step_size * dW1 / (np.sqrt(cache["W1"]) + eps)
      b5 -= step_size * db5 / (np.sqrt(cache["b5"]) + eps)
      b4 -= step_size * db4 / (np.sqrt(cache["b4"]) + eps)
      b3 -= step_size * db3 / (np.sqrt(cache["b3"]) + eps)
      b2 -= step_size * db2 / (np.sqrt(cache["b2"]) + eps)
      b1 -= step_size * db1 / (np.sqrt(cache["b1"]) + eps)

      if i % print_every == 0:
          print("Iter: " + str(i) + " - Mini batch loss: " + str(loss))
          print("Ratio upd W1: " + str(ratio_W1))


  def save_model(self, pk_file):
    self.saved_attr = ["num_input", "num_latent", "num_output", "dropout"]
    self.layers = ["en_hidden", "en_latent_mu", "en_latent_sigma", "de_hidden", "de_out_mu"]
    saved_data = {}

    for attr in self.saved_attr:
      saved_data[attr] = self.__dict__[attr]

    for layer_name in self.layers:
      layer = self.__dict__[attr]
      name = layer_name
      saved_data[name + "_W"]  = layer.W
      saved_data[name + "_b"]  = layer.b

    with open(pk_file, 'wb') as f:
      pk.dump(saved_data, f, pk.HIGHEST_PROTOCOL)



class GauVAE:
  """Gaussian VAE used for real-valued data"""

  def __init__(self, params):
    self.__dict__.update(**params)
    self.num_output = self.num_input
    num_input, num_hidden, dropout, num_latent = self.num_input, self.num_hidden, self.dropout, self.num_latent

    # encoder
    self.en_hidden = LayerFactory.build(self.hidden_activ, num_input, num_output, dropout)  # W1, b1
    self.en_latent_mu = LayerFactory.build("identity", num_hidden, num_latent, dropout)     # W2, b2
    self.en_latent_sigma = LayerFactory.build("exp", num_hidden, num_latent, 1.)            # W3, b3

    # decoder
    self.de_hidden = LayerFactory.build(self.hidden_activ, num_latent, num_hidden, dropout) # W4, b4
    self.de_out_mu = LayerFactory.build("identity", num_hidden, self.num_output, dropout)   # W5, b5
    self.de_out_sigma = LayerFactory.build("exp", num_hidden, self.num_output, 1.)          # W6, b6


  def __call__(self, X_batch, epsilon=None):
    """
    Perform the forward pass

    Inputs:
    - X_batch: A numpy array of shape (num_input, mini_batch) represents the batch training data
    - dropout: (boolean) Whether to activate dropout during forward pass
    - epsilon: A numpy array of shape (num_latent, 1) represents a sample from Standard Normal distribution

    Returns:
    - loss: (scalar) Value of loss function
    """
    num_latent = self.num_latent
    eps = 1e-12
    # encoder
    h_en     = self.en_hidden(X_batch)
    z_mu     = self.en_latent_mu(h_en)
    z_sigma2 = self.en_latent_sigma(h_en)

    # we sample from the posterior q(z|x)
    if epsilon is None: epsilon = np.random.normal(size=num_latent).reshape(num_latent, 1)
    z_samples = z_mu + np.sqrt(z_sigma2) * epsilon

    # decoder
    h_de     = self.de_hidden(z_samples)
    x_mu     = self.de_out_mu(h_de)
    x_sigma2 = self.de_out_sigma(h_de)

    # use the z_samples to estimate the ELBO
    neg_log_gaussian = lambda x, m, s2: 0.5*np.log(2.*np.pi*s2) + (x - m)**2 / (2.*s2)
    neg_kl_divergence = 0.5*(-1. - np.log(z_sigma2) + z_mu**2 + z_sigma2)

    # minimize this loss instead of maximizing
    loss = neg_kl_divergence.sum() + neg_log_gaussian(X_batch, x_mu, x_sigma2).sum()

    self.h_en, self.z_mu, self.z_sigma2, self.epsilon, self.z_samples = h_en, z_mu, z_sigma2, epsilon, z_samples
    self.h_de, self.x_mu, self.x_sigma2, self.X_batch = h_de, x_mu, x_sigma2, X_batch
    return loss


  def generate_data(self, samples=None):
    # sample from the prior
    z_samples = np.random.normal(size=self.num_latent).reshape(self.num_latent, 1) if samples is None else samples

    # forward to the output
    h_de     = self.de_hidden.predict(z_samples)
    x_mu     = self.de_out_mu.predict(h_de)
    return x_mu


  def back_prop(self):
    eps = 1e-12
    h_en, z_mu, z_sigma2, epsilon, z_samples = self.h_en, self.z_mu, self.z_sigma2, self.epsilon, self.z_samples
    h_de, x_mu, x_sigma2, X_batch = self.h_de, self.x_mu, self.x_sigma2, self.X_batch
    dloss_dxm   = (x_mu - X_batch) / x_sigma2
    dloss_dxs   = 0.5/x_sigma2 - 0.5*(X_batch - x_mu)**2 / x_sigma2**2
    dloss_dhde1, dloss_dW5, dloss_db5 = self.de_out_mu.back_prop(dloss_dxm)
    dloss_dhde2, dloss_dW6, dloss_db6 = self.de_out_sigma.back_prop(dloss_dxs)

    dloss_dhde = dloss_dhde1 + dloss_dhde2
    dloss_dz_samples, dloss_dW4, dloss_db4 = self.de_hidden.back_prop(dloss_dhde)

    # loss to zm, zs2
    dloss_dzm1  = z_mu
    dloss_dzs1  = -0.5/z_sigma2 + 0.5

    # propagate from z_samples
    dz_samples_dzm2 = 1.
    dz_samples_dzs2 = 0.5*epsilon * (z_sigma2)**-0.5
    dloss_dzm2 = dloss_dz_samples * dz_samples_dzm2
    dloss_dzs2 = dloss_dz_samples * dz_samples_dzs2

    dloss_dzm = dloss_dzm1 + dloss_dzm2
    dloss_dzs = dloss_dzs1 + dloss_dzs2
    dloss_dhen1, dloss_dW2, dloss_db2 = self.en_latent_mu.back_prop(dloss_dzm)
    dloss_dhen2, dloss_dW3, dloss_db3 = self.en_latent_sigma.back_prop(dloss_dzs)
    dloss_dhen = dloss_dhen1 + dloss_dhen2
    _, dloss_dW1, dloss_db1 = self.en_hidden.back_prop(dloss_dhen)

    return dloss_dW6, dloss_db6, dloss_dW5, dloss_db5, dloss_dW4, dloss_db4, dloss_dW3, dloss_db3, dloss_dW2, dloss_db2, dloss_dW1, dloss_db1


  def train(self, X, batch_size=100, num_iter=1000, step_size=0.001, print_every=100):
    """
    Perform training procedure using Adagrad
    """
    W6, b6, W5, b5, W4, b4 = self.de_out_sigma.W, self.de_out_sigma.b, self.de_out_mu.W, self.de_out_mu.b, self.de_hidden.W, self.de_hidden.b
    W3, b3, W2, b2, W1, b1 = self.en_latent_sigma.W, self.en_latent_sigma.b, self.en_latent_mu.W, self.en_latent_mu.b, self.en_hidden.W, self.en_hidden.b
    eps = 1e-12
    num_train = X.shape[1]
    cache = {"W6": 0., "b6": 0., "W5": 0., "W4": 0., "W3": 0., "W2": 0., "W1": 0., "b5": 0., "b4": 0., "b3": 0., "b2": 0., "b1": 0.}

    for i in range(num_iter+1):
      # create mini-batch
      ix_batch = np.random.choice(range(num_train), size=batch_size, replace=False)
      X_batch  = X[:, ix_batch]

      loss = self.__call__(X_batch)
      dW6, db6, dW5, db5, dW4, db4, dW3, db3, dW2, db2, dW1, db1 = self.back_prop()

      # Adagrad update
      cache["W6"] += dW6**2
      cache["W5"] += dW5**2
      cache["W4"] += dW4**2
      cache["W3"] += dW3**2
      cache["W2"] += dW2**2
      cache["W1"] += dW1**2
      cache["b6"] += db6**2
      cache["b5"] += db5**2
      cache["b4"] += db4**2
      cache["b3"] += db3**2
      cache["b2"] += db2**2
      cache["b1"] += db1**2

      W6 -= step_size * dW6 / (np.sqrt(cache["W6"]) + eps)
      W5 -= step_size * dW5 / (np.sqrt(cache["W5"]) + eps)
      W4 -= step_size * dW4 / (np.sqrt(cache["W4"]) + eps)
      W3 -= step_size * dW3 / (np.sqrt(cache["W3"]) + eps)
      W2 -= step_size * dW2 / (np.sqrt(cache["W2"]) + eps)
      W1 -= step_size * dW1 / (np.sqrt(cache["W1"]) + eps)
      b6 -= step_size * db6 / (np.sqrt(cache["b6"]) + eps)
      b5 -= step_size * db5 / (np.sqrt(cache["b5"]) + eps)
      b4 -= step_size * db4 / (np.sqrt(cache["b4"]) + eps)
      b3 -= step_size * db3 / (np.sqrt(cache["b3"]) + eps)
      b2 -= step_size * db2 / (np.sqrt(cache["b2"]) + eps)
      b1 -= step_size * db1 / (np.sqrt(cache["b1"]) + eps)

      if i % print_every == 0: print("Iter: " + str(i) + " - Mini batch loss: " + str(loss))


  def save_model(self, pk_file):
    self.saved_attr = ["num_input", "num_latent", "num_output", "dropout"]
    self.layers = ["en_hidden", "en_latent_mu", "en_latent_sigma", "de_hidden", "de_out_mu", "de_out_sigma"]
    saved_data = {}

    for attr in self.saved_attr:
      saved_data[attr] = self.__dict__[attr]

    for layer_name in self.layers:
      layer = self.__dict__[attr]
      name = layer_name
      saved_data[name + "_W"]  = layer.W
      saved_data[name + "_b"]  = layer.b

    with open(pk_file, 'wb') as f:
      pk.dump(saved_data, f, pk.HIGHEST_PROTOCOL)

# [TODO] restore model 
