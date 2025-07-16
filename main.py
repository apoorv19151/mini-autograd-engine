import math
import random
import logging
import json
import sys
import os
from typing import List, Optional, Dict, Union, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nn.core import Config, MLP, Trainer, CrossEntropyLoss, Adam, StepLR
from autograd.engine import Value

def create_blobs_dataset(n_samples=1000, n_features=2, n_classes=3, random_state=42):
    """Create a blob dataset perfect for testing neural networks."""

    # Generate the data
    X, y = make_blobs(
        n_samples=n_samples,
        centers=n_classes,
        n_features=n_features,
        random_state=random_state,
        cluster_std=1.5,
        center_box=(-5, 5)
    )

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Normalize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler

def plot_decision_boundary(model, X, y, scaler, title="Decision Boundary"):
    """Plot the decision boundary of the trained model."""

    # Create a mesh to plot the decision boundary
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

    # Get predictions for the mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_scaled = scaler.transform(mesh_points)

    # Predict on mesh points
    predictions = []
    for point in mesh_scaled:
        x_val = [Value(xi) for xi in point]
        y_pred = model(x_val)
        pred_class = max(range(len(y_pred)), key=lambda i: y_pred[i].data)
        predictions.append(pred_class)

    predictions = np.array(predictions).reshape(xx.shape)

    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, predictions, alpha=0.8, cmap=plt.cm.RdYlBu)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

def main():
    """Main training loop for make_blobs dataset."""

    # Create dataset
    print("Creating blobs dataset.")
    X_train, X_test, y_train, y_test, scaler = create_blobs_dataset(
        n_samples=1000,
        n_features=2,
        n_classes=3,
        random_state=42
    )

    print(f"Dataset created:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Classes: {len(np.unique(y_train))}")

    # Plot original data
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.RdYlBu, edgecolors='black')
    plt.title('Training Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    plt.subplot(1, 2, 2)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.RdYlBu, edgecolors='black')
    plt.title('Test Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.tight_layout()
    plt.show()

    # Create configuration
    config = Config(
        learning_rate=0.001,
        batch_size=16,
        epochs=100,
        hidden_sizes=[8, 6],  # Small network: 2 → 8 → 6 → 3
        activation='relu',
        optimizer='adam',
        weight_init='he',
        log_interval=10
    )

    # Build model
    print("\nBuilding model.")
    model = MLP(
        nin=2,  # 2 input features
        architecture=config.hidden_sizes + [3],  # 3 output classes
        activations=['relu', 'relu', 'linear'],
        init_method=config.weight_init
    )

    # Loss and optimizer
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        config=config,
        scheduler=scheduler
    )

    # Train the model
    print("\nStarting training.")
    history = trainer.train(
        X_train.tolist(),
        y_train.tolist(),
        X_test.tolist(),
        y_test.tolist()
    )

    # Final evaluation
    print("\nFinal evaluation.")
    final_metrics = trainer.evaluate(X_test.tolist(), y_test.tolist())

    print("\nFinal Test Metrics:")
    for metric, value in final_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Plot decision boundary
    print("\nGenerating decision boundary plot.")
    plot_decision_boundary(model, X_test, y_test, scaler, "Decision Boundary on Test Data")

    return model, history, final_metrics

if __name__ == "__main__":
    model, history, metrics = main()

