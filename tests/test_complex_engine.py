import math
import unittest

import torch

from autograd.engine import Value


class TestComplexAutogradGraph(unittest.TestCase):
    REL_TOL = 1e-9
    ABS_TOL = 1e-12

    def assert_close(self, actual: float, expected: float, message: str) -> None:
        self.assertTrue(
            math.isclose(actual, expected, rel_tol=self.REL_TOL, abs_tol=self.ABS_TOL),
            msg=f"{message}: expected {expected}, got {actual}",
        )

    def test_complex_expression_matches_pytorch(self):
        x = Value(2.0, label="x")
        y = Value(1.0, label="y")

        result = (
            ((x * y + 3.0) ** 2 - y / (x + 1.0)) * (x - y).relu()
            + x.tanh() * y.sigmoid()
            + (-x).exp()
        )

        tx = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        ty = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)

        torch_result = (
            ((tx * ty + 3.0) ** 2 - ty / (tx + 1.0)) * torch.relu(tx - ty)
            + torch.tanh(tx) * torch.sigmoid(ty)
            + torch.exp(-tx)
        )

        result.backward()
        torch_result.backward()

        self.assert_close(result.data, torch_result.item(), "Forward result mismatch")
        self.assert_close(x.grad, tx.grad.item(), "Gradient mismatch for x")
        self.assert_close(y.grad, ty.grad.item(), "Gradient mismatch for y")


if __name__ == "__main__":
    unittest.main()
