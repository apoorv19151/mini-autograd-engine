import random
import logging

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from dataclasses import asdict
import json

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
    )

    # Normalize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler

def plot_decision_boundary(model, X, y, title="Decision Boundary", save_path=None):
    """Plot the decision boundary of the trained model."""

    # Create a mesh to plot the decision boundary
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

    # Get predictions for the mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]

    # Predict on mesh points
    predictions = []
    for point in mesh_points:
        x_val = [Value(xi) for xi in point]
        y_pred = model(x_val)
        pred_class = max(range(len(y_pred)), key=lambda i: y_pred[i].data)
        predictions.append(pred_class)

    predictions = np.array(predictions).reshape(xx.shape)

    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, predictions, alpha=0.8, cmap=plt.cm.RdYlBu)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors="black")
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()

def main():
    """Main training loop for make_blobs dataset."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s"
    )

    random.seed(42)
    np.random.seed(42)

    # Create dataset
    print("Creating blobs dataset.")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = create_blobs_dataset(
        n_samples=1000,
        n_features=2,
        n_classes=3,
        random_state=42
    )

    assert len(X_train) == len(y_train) == 700
    assert len(X_val) == len(y_val) == 150
    assert len(X_test) == len(y_test) == 150

    print(f"Dataset created:")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Classes: {len(np.unique(y_train))}")

    # Plot original data
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.RdYlBu, edgecolors="black")
    plt.title("Training Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    plt.subplot(1, 2, 2)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.RdYlBu, edgecolors="black")
    plt.title("Test Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()
    plt.show()

    # Create configuration
    config = Config(
        learning_rate=0.001,
        batch_size=16,
        epochs=50,
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

    # Learning rate scheduler
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
        X_val.tolist(),
        y_val.tolist()
    )

    # Final evaluation
    print("\nFinal evaluation.")
    final_metrics = trainer.evaluate(X_test.tolist(), y_test.tolist())

    results_dir = Path(__file__).resolve().parent / "results" / "blobs_seed42"
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "seed": 42,
        "dataset": {
            "name": "make_blobs",
            "n_samples": 1000,
            "n_features": 2,
            "n_classes": 3,
            "cluster_std": 1.5,
            "center_box": [-5, 5],
            "random_state": 42,
            "split": [0.70, 0.15, 0.15],
            "split_random_state": 42,
            "scaling": "StandardScaler fitted on training data",
        },
        "config": asdict(config),
        "architecture": model.architecture,
        "activations": model.activations,
        "gradient_clip_max_norm": 1.0,
        "scheduler": {"name": "StepLR", "step_size": 20, "gamma": 0.5},
        "checkpoint_rule": "final epoch",
        "history": history,
        "validation_metrics": trainer.evaluate(
            X_val.tolist(), y_val.tolist()
        ),
        "test_metrics": final_metrics,
    }

    with (results_dir / "metrics.json").open("w") as f:
        json.dump(results, f, indent=2, allow_nan=False)

    print(f"Results saved to {results_dir}")

    print("\nFinal Test Metrics:")
    for metric, value in final_metrics.items():
        print(f"{metric}: {value:.4f}")

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
    plt.savefig(results_dir / "training_curves.png", dpi=150)
    plt.show()

    # Plot decision boundary
    print("\nGenerating decision boundary plot.")
    plot_decision_boundary(model, X_test, y_test, "Decision Boundary on Test Data",
                           save_path=results_dir / "decision_boundary.png")

    return model, history, final_metrics

if __name__ == "__main__":
    model, history, metrics = main()

