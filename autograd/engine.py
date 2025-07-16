import math
from typing import List, Tuple, Optional, Union

class Value:
  """Value class stores a scalar value and its gradient."""

  def __init__(self, data : Union[int, float], _children : Tuple = (), _operator : str = '', label : str = ''):
    self.data = data
    self.grad = 0.0
    self._prev = set(_children)
    self._operator = _operator
    self._backward = lambda : None
    self.label = label

  def __add__(self, other : Union['Value', int, float]) -> 'Value':
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')

    def _backward():
      self.grad += out.grad
      other.grad += out.grad
    out._backward = _backward

    return out

  def __mul__(self, other : Union['Value', int, float]) -> 'Value':
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')

    def _backward():
      self.grad += out.grad * other.data
      other.grad += out.grad * self.data
    out._backward = _backward

    return out

  def exp(self) -> 'Value':
    clipped_data = max(min(self.data, 700), -700) # clipping the value to prevent overflow
    out = Value(math.exp(clipped_data), (self,), 'exp')

    def _backward():
      self.grad += out.grad * out.data
    out._backward = _backward

    return out

  def relu(self) -> 'Value':
    out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

    def _backward():
      self.grad += (0 if self.data < 0 else 1) * out.grad
    out._backward = _backward

    return out

  def sigmoid(self) -> 'Value':
    # Adding numerical stability for sigmoid
    if self.data >= 0:
      exp_neg = math.exp(-self.data)
      sig = 1 / (1 + exp_neg)
    else:
      exp_pos = math.exp(self.data)
      sig = exp_pos / (1 + exp_pos)

    out = Value(sig, (self,), 'sigmoid')

    def _backward():
      self.grad += (sig * (1 - sig)) * out.grad
    out._backward = _backward

    return out

  def tanh(self) -> 'Value':
    # Adding numerical stability for tanh
    if self.data > 20:
      t = 1.0
    elif self.data < -20:
      t = -1.0
    else:
      exp_2x = math.exp(2 * self.data)
      t = (exp_2x - 1) / (exp_2x + 1)

    out = Value(t, (self,), 'tanh')

    def _backward():
      self.grad += (1 - t**2) * out.grad
    out._backward = _backward

    return out

  def log(self) -> 'Value':
    # Natural logarithm with numerical stability
    safe_data = max(self.data, 1e-15)
    out = Value(math.log(safe_data), (self,), 'log')

    def _backward():
      self.grad += out.grad / safe_data
    out._backward = _backward

    return out

  def __pow__(self, other : Union[int, float]) -> 'Value':
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
    # Iterative topological sort to avoid recursion depth issues
    visited = set()
    topo = []

    # Use a stack for iterative DFS
    stack = [self]

    while stack:
      node = stack[-1]

      if node in visited:
        stack.pop()
        continue

      # Check if all children have been processed
      all_children_processed = True
      for child in node._prev:
        if child not in visited:
          stack.append(child)
          all_children_processed = False

      # If all children are processed, we can add this node to topo order
      if all_children_processed:
        visited.add(node)
        topo.append(node)
        stack.pop()

    # Run backward pass in reverse topological order
    self.grad = 1.0
    for node in reversed(topo):
      node._backward()

