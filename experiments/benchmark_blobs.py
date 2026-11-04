import json
import platform
from pathlib import Path
from dataclasses import asdict

import random

import numpy as np
import torch
import sklearn

from time import perf_counter
from autograd.engine import Value
from main import create_blobs_dataset
from nn.core import MLP, Config, Trainer, CrossEntropyLoss, Adam, StepLR, Metrics

SEEDS = (42, 43, 44)

def make_models(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)

    model = MLP(
        nin=2,
        architecture=[8, 6, 3],
        activations=["relu", "relu", "linear"],
        init_method="he",
    )

    torch_model = torch.nn.Sequential(
        torch.nn.Linear(2, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 6),
        torch.nn.ReLU(),
        torch.nn.Linear(6, 3),
    ).double()

    return model, torch_model

def copy_weights(model, torch_model):
    pairs = [
        (model.layers[0], torch_model[0]),
        (model.layers[1], torch_model[2]),
        (model.layers[2], torch_model[4]),
    ]

    with torch.no_grad():
        for custom_layer, torch_layer in pairs:
            weights = [
                [weight.data for weight in neuron.w]
                for neuron in custom_layer.neurons
            ]
            biases = [
                neuron.b.data for neuron in custom_layer.neurons
            ]

            torch_layer.weight.copy_(
                torch.tensor(weights, dtype=torch.float64)
            )
            torch_layer.bias.copy_(
                torch.tensor(biases, dtype=torch.float64)
            )

def train_custom(model, X_train, y_train, X_val, y_val, seed):
    config = Config(
        learning_rate=0.001,
        batch_size=16,
        epochs=50,
        hidden_sizes=[8, 6],
        activation="relu",
        optimizer="adam",
        weight_init="he",
        log_interval=100,
    )

    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

    trainer = Trainer(
        model, optimizer, CrossEntropyLoss(), config, scheduler
    )

    # Separate training-order randomness from initialization.
    random.seed(seed)

    X_train_list = X_train.tolist()
    y_train_list = y_train.tolist()
    X_val_list = X_val.tolist()
    y_val_list = y_val.tolist()

    start = perf_counter()
    history = trainer.train(
        X_train_list,
        y_train_list,
        X_val_list,
        y_val_list,
    )
    elapsed = perf_counter() - start

    validation = trainer.evaluate(
        X_val.tolist(), y_val.tolist()
    )

    return trainer, history, validation, elapsed

def train_torch(model, X_train, y_train, X_val, y_val, seed):
    X = torch.tensor(X_train, dtype=torch.float64)
    y = torch.tensor(y_train, dtype=torch.long)
    Xv = torch.tensor(X_val, dtype=torch.float64)
    yv = torch.tensor(y_val, dtype=torch.long)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Matches random.seed(seed) in train_custom without global side effects.
    rng = random.Random(seed)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    start = perf_counter()

    for epoch in range(50):
        model.train()

        # Trainer creates a fresh index list every epoch too.
        indices = list(range(len(X)))
        rng.shuffle(indices)

        total_loss = 0.0
        correct = 0

        for offset in range(0, len(indices), 16):
            batch_indices = indices[offset:offset + 16]
            xb = X[batch_indices]
            yb = y[batch_indices]

            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)

            total_loss += loss.item() * len(batch_indices)
            correct += (logits.argmax(dim=1) == yb).sum().item()

            loss.backward()

            # Match your custom clip_gradient rule.
            parameters = list(model.parameters())
            norm = torch.sqrt(sum(
                p.grad.square().sum() for p in parameters
            ))

            if norm.item() > 1.0:
                coefficient = 1.0 / (norm.item() + 1e-6)
                for p in parameters:
                    p.grad.mul_(coefficient)

            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(Xv)
            val_loss = loss_fn(val_logits, yv).item()
            val_accuracy = (
                (val_logits.argmax(dim=1) == yv)
                .double().mean().item()
            )

        history["train_loss"].append(total_loss / len(X))
        history["train_acc"].append(correct / len(X))
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_accuracy)

        # Same end-of-epoch timing as your custom scheduler.
        if (epoch + 1) % 20 == 0:
            for group in optimizer.param_groups:
                group["lr"] *= 0.5

    elapsed = perf_counter() - start

    validation = {
        "loss": history["val_loss"][-1],
        "accuracy": history["val_acc"][-1],
    }

    return history, validation, elapsed

def evaluate_test(trainer, torch_model, X_test, y_test):
    custom_metrics = trainer.evaluate(
        X_test.tolist(), y_test.tolist()
    )

    custom_logits = np.array([
        [
            output.data
            for output in trainer.model(
                [Value(float(v)) for v in x]
            )
        ]
        for x in X_test
    ])

    torch_model.eval()
    with torch.no_grad():
        torch_logits = torch_model(
            torch.tensor(X_test, dtype=torch.float64)
        )

        torch_loss = torch.nn.functional.cross_entropy(
            torch_logits,
            torch.tensor(y_test, dtype=torch.long),
        ).item()

        torch_predictions = torch_logits.argmax(dim=1).tolist()
        torch_logits_array = torch_logits.numpy()

    labels = y_test.tolist()
    num_classes = trainer.model.architecture[-1]

    torch_metrics = {
        "loss": torch_loss,
        "accuracy": Metrics.accuracy(torch_predictions, labels),
        **Metrics.precision_recall_f1(
            torch_predictions, labels, num_classes
        ),
    }

    custom_predictions = custom_logits.argmax(axis=1)

    comparison = {
        "max_abs_logit_error": float(np.max(
            np.abs(custom_logits - torch_logits_array)
        )),
        "prediction_disagreements": int(np.count_nonzero(
            custom_predictions != np.array(torch_predictions)
        )),
    }

    return custom_metrics, torch_metrics, comparison

def main():
    # Dataset generated exactly once
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = create_blobs_dataset(
        n_samples=1000,
        n_features=2,
        n_classes=3,
        random_state=42,
    )

    x = X_train[0]

    results_dir = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "benchmark_blobs"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    runs = []

    for seed in SEEDS:
        model, torch_model = make_models(seed)
        copy_weights(model, torch_model)

        custom_logits = np.array(
            [output.data for output in model(
                [Value(float(v)) for v in x]
            )],
            dtype=np.float64,
        )

        with torch.no_grad():
            torch_logits = torch_model(
                torch.tensor(x, dtype=torch.float64)
            ).numpy()

        max_abs_error = np.max(
            np.abs(custom_logits - torch_logits)
        )

        print(
            f"Seed {seed}: maximum absolute output error = "
            f"{max_abs_error:.3e}"
        )

        assert max_abs_error < 1e-10, (
            f"Seed {seed}: initial forward pass differs from PyTorch"
        )

        trainer, history, validation, elapsed = train_custom(
            model, X_train, y_train, X_val, y_val, seed
        )

        print(
            f"Seed {seed}: custom validation accuracy="
            f"{validation['accuracy']:.4f}, "
            f"loss={validation['loss']:.4f}, "
            f"training+validation time={elapsed:.2f}s"
        )

        torch_history, torch_validation, torch_elapsed = train_torch(
            torch_model, X_train, y_train, X_val, y_val, seed
        )

        print(
            f"Seed {seed}: PyTorch validation accuracy="
            f"{torch_validation['accuracy']:.4f}, "
            f"loss={torch_validation['loss']:.4f}, "
            f"training+validation time={torch_elapsed:.2f}s"
        )

        max_val_loss_gap = max(
            abs(a - b)
            for a, b in zip(
                history["val_loss"],
                torch_history["val_loss"],
            )
        )

        print(
            f"Seed {seed}: maximum validation-loss gap "
            f"across epochs={max_val_loss_gap:.3e}"
        )

        custom_test, torch_test, comparison = evaluate_test(
            trainer, torch_model, X_test, y_test
        )

        run = {
            "seed": seed,
            "config": asdict(trainer.config),
            "custom": {
                "history": history,
                "validation": validation,
                "test": custom_test,
                "training_validation_seconds": elapsed,
            },
            "pytorch": {
                "history": torch_history,
                "validation": torch_validation,
                "test": torch_test,
                "training_validation_seconds": torch_elapsed,
            },
            "comparison": {
                "initial_max_abs_output_error": float(max_abs_error),
                "max_abs_validation_loss_gap": float(max_val_loss_gap),
                **comparison,
            },
        }

        runs.append(run)

        with (results_dir / f"seed_{seed}.json").open("w") as f:
            json.dump(run, f, indent=2, allow_nan=False)

        print(
            f"Seed {seed}: test accuracy "
            f"custom={custom_test['accuracy']:.4f}, "
            f"PyTorch={torch_test['accuracy']:.4f}; "
            f"prediction disagreements="
            f"{comparison['prediction_disagreements']}"
        )

    summary = {}

    for implementation in ("custom", "pytorch"):
        summary[implementation] = {}

        for metric in ("accuracy", "f1", "loss"):
            values = [
                run[implementation]["test"][metric]
                for run in runs
            ]

            summary[implementation][metric] = {
                "mean": float(np.mean(values)),
                "sample_std": float(np.std(values, ddof=1)),
            }

        times = [
            run[implementation]["training_validation_seconds"]
            for run in runs
        ]

        summary[implementation]["training_validation_seconds"] = {
            "mean": float(np.mean(times)),
            "sample_std": float(np.std(times, ddof=1)),
        }

    report = {
        "settings": {
            "seeds": list(SEEDS),
            "dataset_seed": 42,
            "split_seed": 42,
            "dataset": "make_blobs",
            "n_samples": 1000,
            "n_features": 2,
            "n_classes": 3,
            "cluster_std": 1.5,
            "center_box": [-5, 5],
            "split_sizes": [
                len(X_train), len(X_val), len(X_test)
            ],
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "architecture": [2, 8, 6, 3],
            "activations": ["relu", "relu", "linear"],
            "dtype": "float64",
            "device": "cpu",
            "optimizer": {
                "name": "Adam",
                "lr": 0.001,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.0,
            },
            "gradient_clip_max_norm": 1.0,
            "gradient_clip_epsilon": 1e-6,
            "scheduler_step_size": 20,
            "scheduler_gamma": 0.5,
            "checkpoint_rule": "final epoch",
            "timing_scope": (
                "Training and per-epoch validation; "
                "excludes data conversion and final test evaluation"
            ),
            "timing_repetitions_per_seed": 1,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "numpy": np.__version__,
            "pytorch": str(torch.__version__),
            "scikit_learn": sklearn.__version__,
            "torch_num_threads": torch.get_num_threads(),
        },
        "summary": summary,
        "runs": runs,
    }

    with (results_dir / "summary.json").open("w") as f:
        json.dump(report, f, indent=2, allow_nan=False)

    for implementation in ("custom", "pytorch"):
        accuracy = summary[implementation]["accuracy"]

        print(
            f"{implementation}: test accuracy "
            f"{100 * accuracy['mean']:.2f}% "
            f"+/- {100 * accuracy['sample_std']:.2f} "
            f"percentage points"
        )

    print(f"Benchmark results saved to {results_dir}")

if __name__ == "__main__":
    main()

