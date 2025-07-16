import sys
import os
import logging
import math
import random
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import Value from autograd
from autograd.engine import Value

# Configure logging
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Config:
  """Config class for hyperparameters."""
  learning_rate : float = 0.001
  batch_size : int = 32
  epochs : int = 10
  hidden_sizes : List[int] = None
  activation : str = 'relu'
  optimizer : str = 'adam'
  weight_init : str = 'he'
  log_interval : int = 100

  def __post_init__(self):
    if self.hidden_sizes is None:
      self.hidden_sizes = [128, 64]

  @classmethod
  def from_json(cls, json_path : str) -> 'Config':
    """Load the configuration from JSON file."""
    with open(json_path, 'r') as f:
      config_dict = json.load(f)
    return cls(**config_dict)

# Utility functions
def one_hot(y : int, num_classes : int) -> List[Value]:
  """Create one-hot encoded vector."""
  return [Value(1.0) if i == y else Value(0.0) for i in range(num_classes)]

def clip_gradient(parameters : List[Value], max_norm : float = 1.0):
  """Clip gradients to prevent exploding gradients."""
  total_norm = math.sqrt(sum(p.grad**2 for p in parameters))
  if total_norm > max_norm:
    clip_coef = max_norm / (total_norm + 1e-6)
    for p in parameters:
      p.grad *= clip_coef

def predictions_from_outputs(outputs : List[List[Value]]) -> List[int]:
  """Convert the model outputs to predicted class indices."""
  predictions = []
  for output in outputs:
    pred_class = max(range(len(output)), key = lambda i : output[i].data) # returns the index i with the highest score
    predictions.append(pred_class)
  return predictions

# Weight Initialization
class WeightInitializer:

  @staticmethod
  def xavier_uniform(nin : int, nout : int) -> float:
    limit = math.sqrt(6.0 / (nin + nout))
    return random.uniform(-limit, limit)

  @staticmethod
  def he_uniform(nin : int) -> float:
    limit = math.sqrt(6.0 / nin)
    return random.uniform(-limit, limit)

  @staticmethod
  def he_normal(nin : int) -> float:
    std = math.sqrt(2.0 / nin)
    return random.gauss(0, std)

  @staticmethod
  def normal(mean : float = 0.0, std : float = 0.01) -> float:
    return random.gauss(mean, std)

# Neural Network Component
class Module(ABC):
  """Base class for all neural network modules."""

  @abstractmethod
  def __call__(self, x : List[Value]) -> Union[Value, List[Value]]:
    pass

  @abstractmethod
  def parameters(self) -> List[Value]:
    pass

  def zero_grad(self):
    """Zero out gradients of all parameters."""
    for p in self.parameters():
      p.grad = 0.0

class Neuron:
  def __init__(self, nin : int, activation : str = 'relu', init_method : str = 'he'):
    self.nin = nin
    self.activation = activation

    if init_method == 'he':
      self.w = [Value(WeightInitializer.he_uniform(nin)) for _ in range(nin)]
    elif init_method == 'xavier':
      self.w = [Value(WeightInitializer.xavier_uniform(nin)) for _ in range(nin)]
    elif init_method == 'he_normal':
      self.w = [Value(WeightInitializer.he_normal(nin)) for _ in range(nin)]
    else:
      self.w = [Value(WeightInitializer.normal()) for _ in range(nin)]

    self.b = Value(0.0)

  def __call__(self, x : List[Value]) -> Value:
    assert len(x) == self.nin, f"Expected {self.nin} inputs, got {len(x)}"

    # Linear transformation
    act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)

    # Apply activation
    if self.activation == 'relu':
      return act.relu()
    elif self.activation == 'tanh':
      return act.tanh()
    elif self.activation == 'sigmoid':
      return act.sigmoid()
    elif self.activation == 'linear':
      return act
    else:
      raise ValueError(f"Unknown activation: {self.activation}")

  def parameters(self) -> List[Value]:
    return self.w + [self.b]

class Layer(Module):
  def __init__(self, nin : int, nout : int, activation : str = 'relu', init_method : str = 'he'):
    self.nin = nin
    self.nout = nout
    self.neurons = [Neuron(nin, activation, init_method) for _ in range(nout)]

  def __call__(self, x : List[Value]) -> List[Value]:
    outputs = [neuron(x) for neuron in self.neurons]
    return outputs

  def parameters(self) -> List[Value]:
    return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP(Module):
  def __init__(self, nin : int, architecture : List[int], activations : Optional[List[str]] = None, init_method : str = 'he'):
    self.nin = nin
    self.architecture = architecture

    if activations is None:
      activations = ['relu'] * (len(architecture) - 1) + ['linear']
    self.activations = activations

    assert len(activations) == len(architecture), "Number of activations must match the number of layers"

    layer_sizes = [nin] + architecture
    self.layers = []

    for i in range(len(architecture)):
      layer = Layer(layer_sizes[i], layer_sizes[i+1], activations[i], init_method)
      self.layers.append(layer)

    logger.info(f"Created MLP with architecture: {nin} -> {' -> '.join(map(str, architecture))}")

  def __call__(self, x : List[Value]) -> List[Value]:
    for layer in self.layers:
      x = layer(x)
    return x

  def parameters(self) -> List[Value]:
    return [p for layer in self.layers for p in layer.parameters()]

# Loss Functions
class Loss(ABC):
  """Base class for loss functions."""

  @abstractmethod
  def __call__(self, y_pred : List[Value], y_true : List[Value]) -> Value:
    pass

class MSELoss(Loss):
  """Mean Squared Error loss."""

  def __call__(self, y_pred : List[Value], y_true : List[Value]) -> Value:
    assert len(y_pred) == len(y_true), "Prediction and target size mismatch"

    return sum((yp - yt)**2 for yp, yt in zip(y_pred, y_true))/len(y_pred)

class CrossEntropyLoss(Loss):
  """Cross-entropy loss with numerical stability."""

  def __call__(self, y_pred : List[Value], y_true : List[Value]) -> Value:
    assert len(y_pred) == len(y_true), "Prediction and target size mismatch"

    # Using Log-softmax for numerical stability
    max_val = max(yp.data for yp in y_pred)
    shifted_logits = [yp - max_val for yp in y_pred]
    log_sum_exp = max_val + sum(shifted.exp() for shifted in shifted_logits).log()

    log_probs = [yp - log_sum_exp for yp in y_pred]

    # Cross-entropy loss
    loss = Value(0.0)
    for log_prob, target in zip(log_probs, y_true):
      loss = loss + (target * log_prob * -1)

    return loss

class BinaryCrossEntropyLoss(Loss):
  """Binary Cross-entropy loss with numerical stability."""

  def __call__(self, y_pred : List[Value], y_true : List[Value]) -> Value:
    assert len(y_pred) == len(y_true), "Prediction and target size mismatch"

    loss = Value(0.0)
    for yp, yt in zip(y_pred, y_true):
      sig = yp.sigmoid()
      log_sig = yp - (Value(1.0) + yp.exp()).log()
      log_one_minus_sig = (-yp) - (Value(1.0) + (-yp).exp()).log()

      loss = loss + (yt * log_sig * -1) + ((Value(1.0) - yt) * log_one_minus_sig * -1)

    return loss / len(y_pred)

class Optimizer(ABC):
  """Base class for optimizers."""

  def __init__(self, parameters : List[Value]):
    self.parameters = parameters

  def zero_grad(self):
    for p in self.parameters:
      p.grad = 0.0

  @abstractmethod
  def step(self):
    pass

class SGD(Optimizer):
  """Stochastic Gradient Descent with momentum."""

  def __init__(self, parameters : List[Value], lr : float = 0.01, momentum : float = 0.0, weight_decay : float = 0.0):
    super().__init__(parameters)
    self.lr = lr
    self.momentum = momentum
    self.weight_decay = weight_decay
    self.velocity = [0.0] * len(parameters)

  def step(self):
    for i, p in enumerate(self.parameters):
      if self.weight_decay > 0: # Weight decay (L2 regularization)
        p.grad += self.weight_decay * p.data

      # Momentum
      self.velocity[i] = self.momentum * self.velocity[i] + p.grad
      p.data -= self.lr * self.velocity[i]

class Adam(Optimizer):
  """Adam optimizer with bias correction."""

  def __init__(self, parameters : List[Value], lr : float = 0.001, beta1 : float = 0.9, beta2 : float = 0.999, eps : float = 1e-8, weight_decay : float = 0.0):
    super().__init__(parameters)
    self.lr = lr
    self.beta1 = beta1
    self.beta2 = beta2
    self.eps = eps
    self.weight_decay = weight_decay
    self.t = 0
    self.m = [0.0] * len(parameters) # exponentially weighted average of gradients (momentum)
    self.v = [0.0] * len(parameters) # exponentially weighted average of squared gradients (RMS)

  def step(self):
    self.t += 1

    for i, p in enumerate(self.parameters):
      if self.weight_decay > 0: # Weight decay (L2 regularization)
        p.grad += self.weight_decay * p.data

      # Update biased first moment estimate
      self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad

      # Update biased second raw moment estimate
      self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (p.grad ** 2)

      # Compute bias-corrected first moment estimate
      m_hat = self.m[i] / (1 - self.beta1 ** self.t)

      # Compute bias-corrected second raw moment estimate
      v_hat = self.v[i] / (1 - self.beta2 ** self.t)

      # Update parameters
      p.data -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)

# Learning rate schedulers
class LRScheduler(ABC):
  """Base class for learning rate scheduler."""

  def __init__(self, optimizer : Optimizer):
    self.optimizer = optimizer
    self.base_lr = optimizer.lr

  @abstractmethod
  def step(self, epoch : int):
    pass

class StepLR(LRScheduler):
  """Step learning rate scheduler."""

  def __init__(self, optimizer : Optimizer, step_size : int, gamma : float = 0.1):
    super().__init__(optimizer)
    self.step_size = step_size
    self.gamma = gamma

  def step(self, epoch : int):
    if epoch > 0 and epoch % self.step_size == 0:
      self.optimizer.lr *= self.gamma
      logger.info(f"Learning rate reduced to {self.optimizer.lr}")

# Metrics
class Metrics:
  """Collection of evaluation metrics."""

  @staticmethod
  def accuracy(y_pred : List[int], y_true : List[int]) -> float:
    assert len(y_pred) == len(y_true), "Prediction and target size mismatch"

    correct = sum(1 for pred, true in zip(y_pred, y_true) if pred == true)
    return correct / len(y_pred)

  @staticmethod
  def precision_recall_f1(y_pred : List[int], y_true : List[int], num_classes : int) -> Dict[str, float]:
    assert len(y_pred) == len(y_true), "Prediction and target size mismatch"

    confusion = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

    for pred, true in zip(y_pred, y_true):
      if 0 <= pred < num_classes and 0 <= true < num_classes:
        confusion[true][pred] += 1

    precision = []
    recall = []

    for i in range(num_classes):
      tp = confusion[i][i]
      fp = sum(confusion[j][i] for j in range(num_classes)) - tp
      fn = sum(confusion[i][j] for j in range(num_classes)) - tp

      prec = tp / (tp + fp) if (tp + fp) > 0 else 0
      rec = tp / (tp + fn) if (tp + fn) > 0 else 0

      precision.append(prec)
      recall.append(rec)

    average_precision = sum(precision) / len(precision)
    average_recall = sum(recall) / len(recall)

    average_f1 = (2 * average_precision * average_recall) / (average_precision + average_recall) if (average_precision + average_recall) > 0 else 0

    return {'precision' : average_precision, 'recall' : average_recall, 'f1' : average_f1}

# Training utilities
class Trainer:
  """Memory-efficient training class."""

  def __init__(self, model : MLP, optimizer : Optimizer, loss_fn : Loss, config : Config, scheduler : Optional[LRScheduler] = None):
    self.model = model
    self.optimizer = optimizer
    self.loss_fn = loss_fn
    self.config = config
    self.scheduler = scheduler
    self.history = {'train_loss' : [], 'train_acc' : [], 'val_loss' : [], 'val_acc' : []}

  def train_epoch(self, X_train : List[List[float]], y_train : List[int]) -> Tuple[float, float]:
    """Train for one epoch with memory-efficient processing."""
    total_loss = 0.0
    all_predictions = []
    n_samples = len(X_train)

    # Process samples individually to avoid huge computational graphs
    for i in range(0, n_samples, self.config.batch_size):
      batch_end = min(i + self.config.batch_size, n_samples)
      batch_size = batch_end - i

      # Zero gradients for this batch
      self.optimizer.zero_grad()

      batch_loss = 0.0
      batch_predictions = []

      # Process each sample in the batch
      for j in range(i, batch_end):
        x, y = X_train[j], y_train[j]

        # Forward pass
        x_val = [Value(xi) for xi in x]
        y_pred = self.model(x_val)

        # Get prediction for accuracy calculation
        pred_class = max(range(len(y_pred)), key=lambda k: y_pred[k].data)
        batch_predictions.append(pred_class)

        # Calculate loss
        y_true = one_hot(y, len(y_pred))
        loss = self.loss_fn(y_pred, y_true)

        batch_loss += loss.data

        # Backward pass for this sample
        loss.backward() 

      # Now scaling the gradients by batch size (averaging them)
      for param in self.model.parameters():
        param.grad /= batch_size

      # Clip gradients
      clip_gradient(self.model.parameters(), max_norm=1.0)

      # Update parameters
      self.optimizer.step()

      # For logging, using average loss
      avg_batch_loss = batch_loss / batch_size
      total_loss += avg_batch_loss
      all_predictions.extend(batch_predictions)

      # Logging
      if (i // self.config.batch_size) % self.config.log_interval == 0:
        logger.info(f"Batch {i//self.config.batch_size + 1}/{(n_samples-1)//self.config.batch_size + 1}, Loss: {avg_batch_loss:.4f}")

    avg_loss = total_loss / ((n_samples - 1) // self.config.batch_size + 1)

    # Calculate accuracy
    actual_labels = y_train[:len(all_predictions)]
    accuracy = Metrics.accuracy(all_predictions, actual_labels)

    return avg_loss, accuracy

  def evaluate(self, X_test : List[List[float]], y_test : List[int]) -> Dict[str, float]:
    """Evaluate the model on test set."""
    predictions = []
    total_loss = 0.0

    for x, y in zip(X_test, y_test):
      # Forward pass
      x_val = [Value(xi) for xi in x]
      y_pred = self.model(x_val)

      # Get prediction
      pred_class = max(range(len(y_pred)), key=lambda k: y_pred[k].data)
      predictions.append(pred_class)

      # Calculate loss
      y_true = one_hot(y, len(y_pred))
      loss = self.loss_fn(y_pred, y_true)
      total_loss += loss.data

    avg_loss = total_loss / len(X_test)

    # Calculate metrics
    accuracy = Metrics.accuracy(predictions, y_test)
    num_classes = len(set(y_test))
    detailed_metrics = Metrics.precision_recall_f1(predictions, y_test, num_classes)

    return {
      'loss': avg_loss,
      'accuracy': accuracy,
      'precision': detailed_metrics['precision'],
      'recall': detailed_metrics['recall'],
      'f1': detailed_metrics['f1']
    }

  def train(self, X_train : List[List[float]], y_train : List[int], X_val : Optional[List[List[float]]] = None, y_val : Optional[List[int]] = None) -> Dict[str, List[float]]:
    """Memory-efficient training loop."""

    logger.info("Starting training.")
    logger.info(f"Config: {self.config}")

    best_val_acc = 0.0

    for epoch in range(self.config.epochs):
      logger.info(f"\nEpoch {epoch + 1}/{self.config.epochs}")

      # Training
      train_loss, train_acc = self.train_epoch(X_train, y_train)
      self.history['train_loss'].append(train_loss)
      self.history['train_acc'].append(train_acc)

      # Validation
      if X_val is not None and y_val is not None:
        val_metrics = self.evaluate(X_val, y_val)
        self.history['val_loss'].append(val_metrics['loss'])
        self.history['val_acc'].append(val_metrics['accuracy'])

        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        logger.info(f"Val Precision: {val_metrics['precision']:.4f}, Val Recall: {val_metrics['recall']:.4f}, Val F1: {val_metrics['f1']:.4f}")

        if val_metrics['accuracy'] > best_val_acc:
          best_val_acc = val_metrics['accuracy']
          logger.info(f"New best validation accuracy: {best_val_acc:.4f}")
      else:
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

      # Learning rate scheduling
      if self.scheduler:
        self.scheduler.step(epoch)

    logger.info("Training completed!")
    return self.history

