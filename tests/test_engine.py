import math
from tensor_engine import Tensor, Linear, MLP, SGD

def test_strides_and_indexing():
    # 2x3 tensor: shape (2, 3) -> strides [3, 1]
    t = Tensor([[1.0, 2.0, 3.0], 
                [4.0, 5.0, 6.0]])
    assert t.shape == (2, 3)
    assert t.strides == [3, 1]
    assert t[0, 0] == 1.0
    assert t[0, 2] == 3.0
    assert t[1, 1] == 5.0

def test_add_broadcasting_forward_and_backward():
    # (2, 2) + (1, 2) -> (2, 2)
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[10.0, 20.0]])
    c = a + b
    assert c.shape == (2, 2)
    assert c.data == [11.0, 22.0, 13.0, 24.0]
    
    # Backward pass
    loss = c.sum()
    loss.backwards()
    assert a.grad.data == [1.0, 1.0, 1.0, 1.0]
    # b is broadcasted across 2 rows, so its gradient should be summed: [2.0, 2.0]
    assert b.grad.data == [2.0, 2.0]

def test_matmul_forward_and_backward():
    # A (2, 2) @ B (2, 2)
    A = Tensor([[1.0, 2.0], [3.0, 4.0]])
    B = Tensor([[5.0, 6.0], [7.0, 8.0]])
    C = A @ B
    assert C.data == [19.0, 22.0, 43.0, 50.0]
    
    loss = C.sum()
    loss.backwards()
    # out.grad @ B.T = [[1, 1], [1, 1]] @ [[5, 7], [6, 8]] = [[11, 15], [11, 15]]
    assert A.grad.data == [11.0, 15.0, 11.0, 15.0]
    # A.T @ out.grad = [[1, 3], [2, 4]] @ [[1, 1], [1, 1]] = [[4, 4], [6, 6]]
    assert B.grad.data == [4.0, 4.0, 6.0, 6.0]

def test_relu_forward_and_backward():
    x = Tensor([-2.0, 0.0, 3.0])
    y = x.relu(_NegativeSlope=0.01)
    assert abs(y.data[0] - (-0.02)) < 1e-6
    assert abs(y.data[1] - 0.0) < 1e-6
    assert abs(y.data[2] - 3.0) < 1e-6
    
    loss = y.sum()
    loss.backwards()
    assert abs(x.grad.data[0] - 0.01) < 1e-6
    assert abs(x.grad.data[2] - 1.0) < 1e-6

def test_mean_forward_and_backward():
    x = Tensor([2.0, 4.0, 6.0, 8.0])
    m = x.mean()
    assert m.data == [5.0]
    m.backwards()
    # d(mean)/dx_i = 1 / N = 1 / 4 = 0.25
    assert x.grad.data == [0.25, 0.25, 0.25, 0.25]

def test_transpose():
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) # 2x3
    xt = x.T # 3x2
    assert xt.shape == (3, 2)
    assert xt.data == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]

def test_numerical_gradient_check():
    """
    Finite difference check: (f(x + eps) - f(x - eps)) / (2 * eps) == autograd
    """
    eps = 1e-5
    x_val = 3.0
    
    # 1. Autograd gradient of f(x) = x * x + 2 * x
    x = Tensor([x_val])
    f = x * x + Tensor([2.0]) * x
    f.backwards()
    analytical_grad = x.grad.data[0]
    
    # 2. Numerical approximation:
    # f'(x) = 2x + 2 = 2(3) + 2 = 8.0
    f_plus = (x_val + eps) ** 2 + 2.0 * (x_val + eps)
    f_minus = (x_val - eps) ** 2 + 2.0 * (x_val - eps)
    numerical_grad = (f_plus - f_minus) / (2.0 * eps)
    
    assert abs(analytical_grad - numerical_grad) < 1e-4

def test_nn_linear_and_sgd():
    linear = Linear(in_features=2, out_features=1)
    params = list(linear.parameters())
    assert len(params) == 2 # weight and bias
    
    x = Tensor([[1.0, 2.0]])
    out = linear(x)
    assert out.shape == (1, 1)
    
    loss = out.sum()
    loss.backwards()
    
    # Check that gradients were populated
    assert linear.weight.grad is not None
    assert linear.bias.grad is not None
    
    old_weight_val = linear.weight.data[0]
    optimizer = SGD(linear.parameters(), lr=0.1)
    optimizer.step()
    # Weight should have changed
    assert linear.weight.data[0] != old_weight_val
    
    optimizer.zero_grad()
    assert linear.weight.grad is None
    assert linear.bias.grad is None

