import math

class Value:
  """Value class stores a scalar value and its gradient."""
  
  def __init__(self, data, _children=(), _operator='', label=''):
    self.data = data
    self.grad = 0.0
    self._prev = set(_children)
    self._operator = _operator
    self._backward = lambda : None
    self.label = label

  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')

    def _backward():
      self.grad += out.grad
      other.grad += out.grad
    out._backward = _backward

    return out

  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')

    def _backward():
      self.grad += out.grad * other.data
      other.grad += out.grad * self.data
    out._backward = _backward

    return out

  def exp(self):
    out = Value(math.exp(self.data), (self,), 'exp')

    def _backward():
      self.grad += out.grad * out.data
    out._backward = _backward

    return out

  def relu(self):
    out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

    def _backward():
      self.grad += (0 if self.data < 0 else 1) * out.grad
    out._backward = _backward

    return out

  def sigmoid(self):
    x = self.data
    sig = 1/(1 + math.exp(-1*x))
    out = Value(sig, (self,), 'sigmoid')

    def _backward():
      self.grad += (sig * (1 - sig)) * out.grad
    out._backward = _backward

    return out
  
  def tanh(self):
    x = self.data
    t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
    out = Value(t, (self,), 'tanh')

    def _backward():
      self.grad += (1 - t**2) * out.grad
    out._backward = _backward

    return out

  def __pow__(self, other):
    assert isinstance(other, (int, float))
    out = Value(self.data**other, (self,), f'**{other}')

    def _backward():
      self.grad += out.grad * other * self.data**(other - 1)
    out._backward = _backward

    return out

  def __radd__(self, other):
    return self + other

  def __rmul__(self, other):
    return self * other

  def __neg__(self):
    return self * -1

  def __sub__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    return self + (-other)

  def __rsub__(self, other):
    return -self + other

  def __truediv__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    return self * other**-1

  def __rtruediv__(self, other):
    return other * self**-1

  def __repr__(self):
    return f'Value(data={self.data}, prev={self._prev}, label={self.label})'

  def backward(self):
    visited = set()
    topo = []
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)

    self.grad = 1.0
    for node in reversed(topo):
      node._backward()
