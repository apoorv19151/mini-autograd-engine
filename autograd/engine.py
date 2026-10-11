import math
from typing import Tuple, Union

class Value:
  """Value class stores a scalar value and its gradient."""
  __slots__ = (
    "data",
    "grad",
    "_prev",
    "_operator",
    "_backward",
    "label",
  )

  def __init__(self, data: Union[int, float], _children: Tuple = (), _operator: str = '', label: str = ''):
    self.data = float(data)
    self.grad = 0.0
    self._prev = tuple(_children)
    self._operator = _operator
    self._backward = lambda: None
    self.label = label

  def __add__(self, other: Union['Value', int, float]) -> 'Value':
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')

    def _backward():
      self.grad += out.grad
      other.grad += out.grad
    
    out._backward = _backward
    return out

  def __mul__(self, other: Union['Value', int, float]) -> 'Value':
    other = other if isinstance(other, Value) else Value(other)

    left_data = self.data
    right_data = other.data
    out = Value(left_data * right_data, (self, other), '*')

    def _backward():
      self.grad += out.grad * right_data
      other.grad += out.grad * left_data
    
    out._backward = _backward
    return out

  def exp(self) -> 'Value':
    try:
      value = math.exp(self.data)
    except OverflowError:
      value = math.inf

    out = Value(value, (self,), 'exp')

    def _backward():
      self.grad += out.grad * value

    out._backward = _backward
    return out

  def relu(self) -> 'Value':
    out = Value(max(0.0, self.data), (self,), 'ReLU')
    value = self.data

    def _backward():
      self.grad += (1 if value > 0  else 0.0) * out.grad
    
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
    value = math.tanh(self.data)
    out = Value(value, (self,), 'tanh')

    def _backward():
      self.grad += (1 - value**2) * out.grad

    out._backward = _backward
    return out

  def log(self) -> 'Value':
    if self.data <= 0:
      raise ValueError(
        f"log is defined only for positive real values; got {self.data}"
      )

    value = self.data
    out = Value(math.log(value), (self,), 'log')

    def _backward():
      self.grad += out.grad / value

    out._backward = _backward
    return out

  def __pow__(self, exponent: Union[int, float]) -> 'Value':
    if isinstance(exponent, bool) or not isinstance(exponent, (int, float)):
      raise TypeError("exponent must be an int or float")

    base = self.data

    if base == 0 and exponent < 0:
      raise ValueError("zero cannot be raised to a negative power")

    if base < 0 and not float(exponent).is_integer():
      raise ValueError(
        "fractional powers of negative values are not supported"
      )

    out = Value(base**exponent, (self,), f"**{exponent}")

    def _backward():
      if exponent == 0:
        local_gradient = 0.0
      elif base == 0 and exponent == 1:
        local_gradient = 1.0
      elif base == 0 and exponent > 1:
        local_gradient = 0.0
      elif base == 0 and 0 < exponent < 1:
        local_gradient = math.inf
      else:
        local_gradient = exponent * (base ** (exponent - 1))

      self.grad += out.grad * local_gradient

    out._backward = _backward
    return out
  
  def __radd__(self, other: Union['Value', int, float]) -> 'Value':
    return self + other

  def __rmul__(self, other: Union['Value', int, float]) -> 'Value':
    return self * other

  def __neg__(self) -> 'Value':
    return self * -1

  def __sub__(self, other: Union['Value', int, float]) -> 'Value':
    other = other if isinstance(other, Value) else Value(other)
    return self + (-other)

  def __rsub__(self, other: Union['Value', int, float]) -> 'Value':
    return -self + other

  def __truediv__(self, other: Union['Value', int, float]) -> 'Value':
    other = other if isinstance(other, Value) else Value(other)
    return self * other**-1

  def __rtruediv__(self, other: Union['Value', int, float]) -> 'Value':
    return other * self**-1

  def __repr__(self) -> str:
    return (
      f"Value(data={self.data}, grad={self.grad}, op={self._operator!r}, label={self.label!r})"
    )

  def backward(self, gradient: float = 1.0) -> None:
    """Backpropagate and accumulate gradients on leaf values."""
    visited = set()
    topo = []
    stack = [(self, False)]

    while stack:
      node, expanded = stack.pop()

      if node in visited:
        continue

      if expanded:
        visited.add(node)
        topo.append(node)
        continue

      stack.append((node, True))

      for parent in reversed(node._prev):
        if parent not in visited:
          stack.append((parent, False))

    if not self._prev:
      self.grad += float(gradient)
      return

    for node in topo:
      if node._prev:
        node.grad = 0.0

    self.grad = float(gradient)

    for node in reversed(topo):
      node._backward()

