import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import torch
from autograd.engine import Value

class ComplexTestAutogradEngine(unittest.TestCase):

    def test_complex_expression(self):
        x = Value(2.0, label='x')
        y = Value(3.0, label='y')

        # Expression: z = ((x*y + 3)**2 - (y/(x + 1))) * ReLU(x - y) + tanh(x) + sigmoid(y) + exp(-x)
        a = x * y
        b = a + 3
        c = b ** 2
        d = y / (x + 1)
        e = c - d
        f = x - y
        g = f.relu()
        h = e * g
        i = x.tanh()
        j = y.sigmoid()
        k = i * j
        l = (-x).exp()
        z = h + k + l

        # Backward pass
        z.backward()

        # Pytorch implementation
        x_torch = torch.tensor(2.0).double()
        x_torch.requires_grad = True
        y_torch = torch.tensor(3.0).double()
        y_torch.requires_grad = True

        a_torch = x_torch * y_torch
        b_torch = a_torch + 3
        c_torch = b_torch ** 2
        d_torch = y_torch / (x_torch + 1)
        e_torch = c_torch - d_torch
        f_torch = x_torch - y_torch
        g_torch = f_torch.relu()
        h_torch = e_torch * g_torch
        i_torch = x_torch.tanh()
        j_torch = y_torch.sigmoid()
        k_torch = i_torch * j_torch
        l_torch = (-x_torch).exp()
        z_torch = h_torch + k_torch + l_torch

        # Backward pass
        z_torch.backward()

        print(f"Forward pass, z.data = {z.data}, z_torch.item() = {z_torch.item()}")
        self.assertAlmostEqual(z_torch.item(), z.data, places=5)
        print(f"Backward pass, x.grad = {x.grad}, x_torch.grad.item() = {x_torch.grad.item()}")
        self.assertAlmostEqual(x_torch.grad.item(), x.grad, places=5)
        print(f"Backward pass, y.grad = {y.grad}, y_torch.grad.item() = {y_torch.grad.item()}")
        self.assertAlmostEqual(y_torch.grad.item(), y.grad, places=5)

if __name__ == '__main__':
    unittest.main()
