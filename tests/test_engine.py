import unittest
import torch
from autograd.engine import Value

class TestAutogradEngine(unittest.TestCase):

    def _test_operation(self, operation_name, custom_result, torch_result, custom_inputs, torch_inputs):
        print(f"Testing {operation_name}")
        print(f"Custom result: {custom_result.data}, Torch result: {torch_result.item()}")
        custom_result.backward()
        torch_result.backward()
        for c_input, t_input in zip(custom_inputs, torch_inputs):
            print(f"Custom grad: {c_input.grad}, Torch grad: {t_input.grad.item()}")
            self.assertAlmostEqual(c_input.grad, t_input.grad.item(), places=5)

    def test_add(self):
        x, y = Value(3.0), Value(4.0)
        x_tensor = torch.tensor([3.0], requires_grad=True)
        y_tensor = torch.tensor([4.0], requires_grad=True)
        self._test_operation("add", x + y, x_tensor + y_tensor, [x, y], [x_tensor, y_tensor])

    def test_radd(self):
        x = Value(3.0)
        x_tensor = torch.tensor([3.0], requires_grad=True)
        self._test_operation("radd", 4.0 + x, torch.tensor([4.0], requires_grad=True) + x_tensor, [x], [x_tensor])

    def test_sub(self):
        x, y = Value(7.0), Value(4.0)
        x_tensor = torch.tensor([7.0], requires_grad=True)
        y_tensor = torch.tensor([4.0], requires_grad=True)
        self._test_operation("sub", x - y, x_tensor - y_tensor, [x, y], [x_tensor, y_tensor])

    def test_rsub(self):
        x = Value(3.0)
        x_tensor = torch.tensor([3.0], requires_grad=True)
        self._test_operation("rsub", 4.0 - x, torch.tensor([4.0], requires_grad=True) - x_tensor, [x], [x_tensor])

    def test_mul(self):
        x, y = Value(3.0), Value(4.0)
        x_tensor = torch.tensor([3.0], requires_grad=True)
        y_tensor = torch.tensor([4.0], requires_grad=True)
        self._test_operation("mul", x * y, x_tensor * y_tensor, [x, y], [x_tensor, y_tensor])

    def test_rmul(self):
        x = Value(3.0)
        x_tensor = torch.tensor([3.0], requires_grad=True)
        self._test_operation("rmul", 4.0 * x, torch.tensor([4.0], requires_grad=True) * x_tensor, [x], [x_tensor])

    def test_truediv(self):
        x, y = Value(8.0), Value(4.0)
        x_tensor = torch.tensor([8.0], requires_grad=True)
        y_tensor = torch.tensor([4.0], requires_grad=True)
        self._test_operation("truediv", x.__truediv__(y), x_tensor / y_tensor, [x, y], [x_tensor, y_tensor])

    def test_rtruediv(self):
        x = Value(3.0)
        x_tensor = torch.tensor([3.0], requires_grad=True)
        self._test_operation("rtruediv", x.__rtruediv__(4.0), torch.tensor([4.0], requires_grad=True) / x_tensor, [x], [x_tensor])

    def test_neg(self):
        x = Value(3.0)
        x_tensor = torch.tensor([3.0], requires_grad=True)
        self._test_operation("neg", -x, -x_tensor, [x], [x_tensor])

    def test_exp(self):
        x = Value(1.0)
        x_tensor = torch.tensor([1.0], requires_grad=True)
        self._test_operation("exp", x.exp(), torch.exp(x_tensor), [x], [x_tensor])

    def test_relu(self):
        x = Value(3.0)
        x_tensor = torch.tensor([3.0], requires_grad=True)
        self._test_operation("relu", x.relu(), torch.relu(x_tensor), [x], [x_tensor])

    def test_sigmoid(self):
        x = Value(0.5)
        x_tensor = torch.tensor([0.5], requires_grad=True)
        self._test_operation("sigmoid", x.sigmoid(), torch.sigmoid(x_tensor), [x], [x_tensor])

    def test_tanh(self):
        x = Value(0.5)
        x_tensor = torch.tensor([0.5], requires_grad=True)
        self._test_operation("tanh", x.tanh(), torch.tanh(x_tensor), [x], [x_tensor])

    def test_pow(self):
        x = Value(2.0)
        x_tensor = torch.tensor([2.0], requires_grad=True)
        self._test_operation("pow", x ** 3, x_tensor ** 3, [x], [x_tensor])

if __name__ == '__main__':
    unittest.main()
