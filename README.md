# Mini Autograd Engine

A lightweight automatic differentiation engine built from scratch in Python, inspired by PyTorch's autograd system and Andrej Karpathy's micrograd.

## Features

- **Automatic Differentiation**: Forward and backward pass computation for scalar operations
- **Activation Functions**: tanh, sigmoid, ReLU, etc., with their derivatives
- **Neural Network Framework**: Multi-layer perceptron implementation with configurable architecture
- **Training Pipeline**: Complete training loop with optimization, loss functions, and metrics
- **PyTorch Compatibility**: Results verified against PyTorch implementation

## Quick Start

### Installation

```bash
pip install numpy matplotlib scikit-learn graphviz
```

### Basic Usage

```python
from engine import Value

# Create computational graph
x = Value(2.0, label='x')
y = Value(3.0, label='y')

# Complex expression with multiple operations
z = ((x * y + 3)**2 - (y / (x + 1))) * (x - y).relu() + x.tanh() * y.sigmoid() + (-x).exp()

# Compute gradients
z.backward()

print(f"Result: {z.data}")
print(f"Gradients: x.grad={x.grad}, y.grad={y.grad}")
```

### Neural Network Training

```python
# Train on synthetic dataset
python3 main.py
```

## Results

**Test Performance on Make Blobs Dataset:**
- **Accuracy**: 96.0%
- **Loss**: 0.112
- **Precision/Recall/F1**: ~0.96

The engine produces identical results to PyTorch, validating the correctness of the implementation.

## Testing

```bash
# Test individual operations against PyTorch
python3 tests/test_engine.py

# Test complex computational graphs
python3 tests/complex_test_engine.py
```

## Computational Graph Visualization

The engine generates visual representations of computational graphs using Graphviz:

```python
from utils import draw_dot
dot = draw_dot(z)
dot.render('autograd_graph', format='png')
```

## References

- [Andrej Karpathy's micrograd tutorial](https://www.youtube.com/watch?v=VMj-3S1tku0&t=5321s)
- [PyTorch autograd documentation](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html?utm_source=pytorchkr)

## License

MIT License
