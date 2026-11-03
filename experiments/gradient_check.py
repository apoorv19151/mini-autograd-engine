import random
import torch

from autograd.engine import Value
from nn.core import MLP, CrossEntropyLoss, one_hot, SGD, Adam

def run_check(optimizer_name):
    random.seed(42)

    model = MLP(
        nin=2,
        architecture=[3, 2],
        activations=["tanh", "linear"],
        init_method="xavier"
    )

    torch_model = torch.nn.Sequential(
        torch.nn.Linear(2, 3),
        torch.nn.Tanh(),
        torch.nn.Linear(3, 2)
    ).double()

    with torch.no_grad():
        for custom_layer, torch_layer in [
            (model.layers[0], torch_model[0]),
            (model.layers[1], torch_model[2])
        ]:
            for j, neuron in enumerate(custom_layer.neurons):
                weights = [w.data for w in neuron.w]

                torch_layer.weight[j].copy_(
                    torch.tensor(weights, dtype=torch.float64)
                )

                torch_layer.bias[j].copy_(
                    torch.tensor(neuron.b.data, dtype=torch.float64)
                )

    x = [0.25, -0.75]

    custom_output = model([Value(v) for v in x])
    torch_output = torch_model(torch.tensor(x, dtype=torch.float64))

    custom_values = [v.data for v in custom_output]
    torch_values = torch_output.detach().tolist()

    max_error = max(
        abs(a - b) for a, b in zip(custom_values, torch_values)
    )

    print("Custom:", custom_values)
    print("PyTorch:", torch_values)
    print("Maximum absolute error:", max_error)

    assert max_error < 1e-10, "Forward outputs do not match"

    # Compare loss for class 1.
    target = 1

    custom_loss = CrossEntropyLoss()(custom_output, one_hot(target, 2))
    torch_loss = torch.nn.functional.cross_entropy(
        torch_output.unsqueeze(0),
        torch.tensor([target], dtype=torch.long),
    )

    loss_error = abs(custom_loss.data - torch_loss.item())
    print("Loss absolute error:", loss_error)
    assert loss_error < 1e-10, "Loss mismatch"

    # Compute gradients independently.
    model.zero_grad()
    torch_model.zero_grad()

    custom_loss.backward()
    torch_loss.backward()

    # Compare corresponding weights and biases.
    gradient_errors = []

    for layer_index, (custom_layer, torch_layer) in enumerate([
        (model.layers[0], torch_model[0]),
        (model.layers[1], torch_model[2]),
    ]):
        for j, neuron in enumerate(custom_layer.neurons):
            for k, weight in enumerate(neuron.w):
                expected = torch_layer.weight.grad[j, k].item()
                error = abs(weight.grad - expected)
                gradient_errors.append(error)

                assert error < 1e-10, (
                    f"Weight gradient mismatch: layer={layer_index}, "
                    f"neuron={j}, input={k}"
                )

            expected = torch_layer.bias.grad[j].item()
            error = abs(neuron.b.grad - expected)
            gradient_errors.append(error)

            assert error < 1e-10, (
                f"Bias gradient mismatch: layer={layer_index}, neuron={j}"
            )

    print("Parameters checked:", len(gradient_errors))
    print("Maximum absolute gradient error:", max(gradient_errors))

    if optimizer_name == "sgd":
        custom_optimizer = SGD(model.parameters(), lr=0.01)
        torch_optimizer = torch.optim.SGD(torch_model.parameters(), lr=0.01)
    else:
        custom_optimizer = Adam(model.parameters(), lr=0.001)
        torch_optimizer = torch.optim.Adam(torch_model.parameters(), lr=0.001)

    custom_optimizer.step()
    torch_optimizer.step()

    update_errors = []

    for custom_layer, torch_layer in [
        (model.layers[0], torch_model[0]),
        (model.layers[1], torch_model[2]),
    ]:
        for j, neuron in enumerate(custom_layer.neurons):
            for k, weight in enumerate(neuron.w):
                update_errors.append(
                    abs(weight.data - torch_layer.weight[j, k].item())
                )

            update_errors.append(
                abs(neuron.b.data - torch_layer.bias[j].item())
            )

    max_update_error = max(update_errors)

    print(f"Parameters checked after {optimizer_name}:", len(update_errors))
    print(f"Maximum absolute {optimizer_name} parameter error:", max_update_error)

    assert max_update_error < 1e-10, f"{optimizer_name} update mismatch"

if __name__ == "__main__":
    run_check("sgd")
    run_check("adam")
