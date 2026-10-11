import math
import unittest

import torch

from autograd.engine import Value


class TestAutogradEngine(unittest.TestCase):
    REL_TOL = 1e-9
    ABS_TOL = 1e-12

    def assert_close(self, actual: float, expected: float, message: str = "") -> None:
        self.assertTrue(
            math.isclose(actual, expected, rel_tol=self.REL_TOL, abs_tol=self.ABS_TOL),
            msg=f"{message}: expected {expected}, got {actual}",
        )

    @staticmethod
    def torch_value(value: float) -> torch.Tensor:
        return torch.tensor(value, dtype=torch.float64, requires_grad=True)

    def compare_with_torch(
        self,
        custom_result: Value,
        torch_result: torch.Tensor,
        custom_inputs: list[Value],
        torch_inputs: list[torch.Tensor],
    ) -> None:
        self.assertEqual(len(custom_inputs), len(torch_inputs))

        self.assert_close(
            custom_result.data, torch_result.item(), "Forward value mismatch"
        )

        custom_result.backward()
        torch_result.backward()

        for index, (custom_input, torch_input) in enumerate(
            zip(custom_inputs, torch_inputs)
        ):
            self.assertIsNotNone(torch_input.grad)
            self.assert_close(
                custom_input.grad,
                torch_input.grad.item(),
                f"Gradient mismatch for input {index}",
            )

    def test_add(self):
        x = Value(3.0)
        y = Value(4.0)

        tx = self.torch_value(3.0)
        ty = self.torch_value(4.0)

        self.compare_with_torch(x + y, tx + ty, [x, y], [tx, ty])

    def test_radd(self):
        x = Value(3.0)
        tx = self.torch_value(3.0)

        self.compare_with_torch(4.0 + x, 4.0 + tx, [x], [tx])

    def test_sub(self):
        x = Value(7.0)
        y = Value(4.0)

        tx = self.torch_value(7.0)
        ty = self.torch_value(4.0)

        self.compare_with_torch(x - y, tx - ty, [x, y], [tx, ty])

    def test_rsub(self):
        x = Value(7.0)
        tx = self.torch_value(7.0)

        self.compare_with_torch(4.0 - x, 4.0 - tx, [x], [tx])

    def test_mul(self):
        x = Value(3.0)
        y = Value(4.0)

        tx = self.torch_value(3.0)
        ty = self.torch_value(4.0)

        self.compare_with_torch(x * y, tx * ty, [x, y], [tx, ty])

    def test_rmul(self):
        x = Value(3.0)
        tx = self.torch_value(3.0)

        self.compare_with_torch(4.0 * x, 4.0 * tx, [x], [tx])

    def test_truediv(self):
        x = Value(7.0)
        y = Value(4.0)

        tx = self.torch_value(7.0)
        ty = self.torch_value(4.0)

        self.compare_with_torch(x / y, tx / ty, [x, y], [tx, ty])

    def test_rtruediv(self):
        x = Value(3.0)
        tx = self.torch_value(3.0)

        self.compare_with_torch(4.0 / x, 4.0 / tx, [x], [tx])

    def test_neg(self):
        x = Value(3.0)
        tx = self.torch_value(3.0)

        self.compare_with_torch(-x, -tx, [x], [tx])

    def test_exp(self):
        x = Value(3.0)
        tx = self.torch_value(3.0)

        self.compare_with_torch(x.exp(), tx.exp(), [x], [tx])

    def test_relu(self):
        x = Value(3.0)
        tx = self.torch_value(3.0)

        self.compare_with_torch(x.relu(), tx.relu(), [x], [tx])

    def test_sigmoid(self):
        x = Value(3.0)
        tx = self.torch_value(3.0)

        self.compare_with_torch(x.sigmoid(), tx.sigmoid(), [x], [tx])

    def test_tanh(self):
        x = Value(3.0)
        tx = self.torch_value(3.0)

        self.compare_with_torch(x.tanh(), tx.tanh(), [x], [tx])

    def test_pow(self):
        x = Value(3.0)
        tx = self.torch_value(3.0)

        self.compare_with_torch(x**2.0, tx**2.0, [x], [tx])

    def test_negative_base_integer_power(self):
        x = Value(-2.0)
        tx = self.torch_value(-2.0)

        self.compare_with_torch(x**3, tx**3, [x], [tx])

    def test_log(self):
        x = Value(3.0)
        tx = self.torch_value(3.0)

        self.compare_with_torch(x.log(), tx.log(), [x], [tx])

    def test_shared_operand_multiplication(self):
        x = Value(3.0)
        result = x * x

        result.backward()

        self.assert_close(result.data, 9.0)
        self.assert_close(x.grad, 6.0)

    def test_shared_operand_addition(self):
        x = Value(3.0)
        result = x + x

        result.backward()

        self.assert_close(result.data, 6.0)
        self.assert_close(x.grad, 2.0)

    def test_branched_graph(self):
        x = Value(2.0)
        y = Value(3.0)

        result = (x * y) + (x**2)
        result.backward()

        self.assert_close(result.data, 10.0)
        self.assert_close(x.grad, 7.0)
        self.assert_close(y.grad, 2.0)

    def test_repeated_backward_accumulates_leaf_gradients(self):
        x = Value(2.0)
        intermediate = x * x
        result = intermediate * intermediate

        result.backward()
        self.assert_close(x.grad, 32.0)

        result.backward()
        self.assert_close(x.grad, 64.0)

    def test_custom_backward_gradient(self):
        x = Value(3.0)
        result = x * x

        result.backward(gradient=0.5)

        self.assert_close(x.grad, 3.0)

    def test_deep_graph(self):
        x = Value(1.0)
        result = x

        for _ in range(10000):
            result = result + 1.0

        result.backward()

        self.assert_close(result.data, 10001.0)
        self.assert_close(x.grad, 1.0)

    def test_relu_at_zero(self):
        x = Value(0.0)
        result = x.relu()

        result.backward()

        self.assert_close(result.data, 0.0)
        self.assert_close(x.grad, 0.0)

    def test_relu_negative_input(self):
        x = Value(-3.0)
        tx = self.torch_value(-3.0)

        self.compare_with_torch(x.relu(), tx.relu(), [x], [tx])

    def test_log_rejects_zero(self):
        with self.assertRaises(ValueError):
            Value(0.0).log()

    def test_log_rejects_negative_input(self):
        with self.assertRaises(ValueError):
            Value(-1.0).log()

    def test_zero_power_zero(self):
        x = Value(0.0)
        result = x**0

        result.backward()

        self.assert_close(result.data, 1.0)
        self.assert_close(x.grad, 0.0)

    def test_zero_negative_power_is_rejected(self):
        with self.assertRaises(ValueError):
            Value(0.0) ** -1

    def test_negative_fractional_power_is_rejected(self):
        with self.assertRaises(ValueError):
            Value(-2.0) ** 0.5

    def test_boolean_exponent_is_rejected(self):
        with self.assertRaises(TypeError):
            Value(2.0) ** True

    def test_division_by_zero_is_rejected(self):
        with self.assertRaises(ValueError):
            Value(2.0) / Value(0.0)

    def test_sigmoid_large_positive_input(self):
        x = Value(1000.0)
        result = x.sigmoid()

        result.backward()

        self.assertTrue(math.isfinite(result.data))
        self.assert_close(result.data, 1.0)
        self.assert_close(x.grad, 0.0)

    def test_sigmoid_large_negative_input(self):
        x = Value(-1000.0)
        result = x.sigmoid()

        result.backward()

        self.assertTrue(math.isfinite(result.data))
        self.assert_close(result.data, 0.0)
        self.assert_close(x.grad, 0.0)

    def test_multiplication_backward_uses_forward_values(self):
        x = Value(2.0)
        y = Value(3.0)
        result = x * y

        x.data = 100.0
        y.data = 200.0

        result.backward()

        self.assert_close(x.grad, 3.0)
        self.assert_close(y.grad, 2.0)
