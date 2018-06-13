import numpy as np

# We use the following convention for derivatives
# df_dx is partial derivative of f wrt x
#
# Rows are related to f, columns are related to x
# df_dx = [df1_dx1   df1_dx2   ... ]
#         [df2_dx1   df2_dx2   ... ]
#         [...       ...       ... ]
#
# We assume `out` is scalar so dout_dx is [1, dim_x] and dout_dW = shape(W)
#
class IdentityLayer:

  def __init__(self, num_input, num_output, dropout=1.):
    self.num_input  = num_input
    self.num_output = num_output
    self.W, self.b  = self._init_weights_and_biases()
    self.dropout    = dropout
    self.activation = lambda x: x

  def _init_weights_and_biases(self):
    """UPDATE: use zero init for biases and small weight init"""
    W = np.random.normal(size=(self.num_output, self.num_input)) / np.sqrt(self.num_input)
    # W = 0.01 * np.random.normal(size=(self.num_output, self.num_input))
    b = np.zeros(shape=(self.num_output, 1))
    return W, b

  def __call__(self, X, save_input=True):
    """
    Perform the forward pass

    Inputs:
    - X: A numpy array of shape (num_input, num_train)
    - save_input: (boolean) Whether to save X for back_prop

    Returns:
    - Z: A numpy array of shape (num_output, num_train)
    """
    W, b, num_input, dropout = self.W, self.b, self.num_input, self.dropout
    if save_input: self.X = X                  # save the data for back_prop
    self.H = W.dot(X) + b                      # (out, in) x (in, train) + (out, 1) = (out, train)
    Z = self.activation(self.H)
    # randomly zeroing and rescaling output
    self.Z = (np.random.rand(*Z.shape) <= dropout) * Z/dropout
    return self.Z


  def predict(self, X):
    """
    Predict the output given test data X
    """
    W, b = self.W, self.b
    H = W.dot(X) + b                 # (out, in) x (in, train) + (out, 1) = (out, train)
    return self.activation(H)


  def back_prop(self, dout):
    """
    Back propagate the gradient from output to input

    Note: dout means dout_dZ
    """
    W, b, X, Z = self.W, self.b, self.X, self.Z
    num_input, num_output = self.num_input, self.num_output
    dZ_dW = X.T
    dZ_dX = W.T

    dout    = dout * (Z != 0.)
    dout_dW = dout @ dZ_dW                # (out, train) x (train, in) = (out, in)
    dout_db = dout.sum(axis=1)            # (out, )
    dout_dX = dZ_dX @ dout                # (in, out) x (out, train) = (in, train)
    return dout_dX, dout_dW, dout_db.reshape(num_output, 1)


class SigmoidLayer(IdentityLayer):

  def __init__(self, num_input, num_output, dropout=1.):
    super().__init__(num_input, num_output, dropout)
    self.activation = lambda x: 1. / (1. + np.exp(-x))

  def back_prop(self, dout):
    """dout means dout_dz"""
    W, b, X, Z = self.W, self.b, self.X, self.Z
    num_input, num_output = self.num_input, self.num_output
    dH_dW = X.T
    dH_dX = W.T

    dout    = dout * (Z != 0.)
    dZ_dH   = Z * (1. - Z)
    dout_dH = dout * dZ_dH
    dout_dW = dout_dH @ dH_dW
    dout_db = dout_dH.sum(axis=1)
    dout_dX = dH_dX @ dout_dH
    return dout_dX, dout_dW, dout_db.reshape(num_output, 1)


class TanhLayer(IdentityLayer):

  def __init__(self, num_input, num_output, dropout=1.):
    super().__init__(num_input, num_output, dropout)
    self.activation = lambda x: np.tanh(x)

  def back_prop(self, dout):
    """dout means dout_dz"""
    W, b, X, Z = self.W, self.b, self.X, self.Z
    num_input, num_output = self.num_input, self.num_output
    dH_dW = X.T
    dH_dX = W.T

    dout    = dout * (Z != 0.)
    dZ_dH   = 1. - Z**2
    dout_dH = dout * dZ_dH
    dout_dW = dout_dH @ dH_dW
    dout_db = dout_dH.sum(axis=1)
    dout_dX = dH_dX @ dout_dH
    return dout_dX, dout_dW, dout_db.reshape(num_output, 1)


class ExpLayer(IdentityLayer):

  def __init__(self, num_input, num_output, dropout=1.):
    super().__init__(num_input, num_output, dropout)
    self.activation = lambda x: np.exp(x)

  def back_prop(self, dout):
    """dout means dout_dz"""
    W, b, X, Z = self.W, self.b, self.X, self.Z
    num_input, num_output = self.num_input, self.num_output
    dH_dW = X.T
    dH_dX = W.T

    dout    = dout * (Z != 0.)
    dZ_dH   = Z
    dout_dH = dout * dZ_dH
    dout_dW = dout_dH @ dH_dW
    dout_db = dout_dH.sum(axis=1)
    dout_dX = dH_dX @ dout_dH
    return dout_dX, dout_dW, dout_db.reshape(num_output, 1)


# No gradient check yet
class SoftmaxLayer(IdentityLayer):

  def __init__(self, num_input, num_output, dropout=1.):
    super().__init__(num_input, num_output, dropout)
    self.activation = lambda x: np.exp(x) / np.sum(np.exp(x))

  def back_prop(self, dout):
    """dout means dout_dz"""
    W, b, x, z = self.W, self.b, self.x, self.z
    num_input, num_output = self.num_input, self.num_output
    dout  = np.reshape(dout, (1, num_output))

    dout    = dout * (Z != 0.)
    Z = np.repeat(np.reshape(z, (num_output, 1)), num_output, axis=1)  # repeat per column
    I = np.eye(num_output)
    dz_dh = Z * (I - Z) if np.ndim(z) > 0 else z * (1. - z)
    dout_dW = dout.dot(dz_dh).dot(dh_dw).reshape(num_output, num_input)
    dout_db = dout.dot(dz_dh).dot(dh_db)
    dout_dx = dout.dot(dz_dh).dot(dh_dx)
    return dout_dx, dout_dW, dout_db


class LayerFactory:

  @staticmethod
  def build(layer_name, arg):
    if   layer_name == "identity": return IdentityLayer(*arg)
    elif layer_name == "sigmoid":  return SigmoidLayer(*arg)
    elif layer_name == "tanh":     return TanhLayer(*arg)
    elif layer_name == "exp":      return ExpLayer(*arg)
    elif layer_name == "softmax":  return SoftmaxLayer(*arg)
