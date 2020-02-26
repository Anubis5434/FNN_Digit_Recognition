# shows how you can compose different modules,
# and iterate over their parameters.
# this is useful when writing `clear_grad` and `update_param`.

import numpy as np
from hw4_lib import Module


class MyNNLambda(Module):
    # a dummy lambda module that does nothing to the input.
    def __init__(self):
        super().__init__()
        # a weight parameter of shape (5,2)
        self._register_param('weight', np.arange(10.0).reshape(5, 2))
        # a bias parameter of shape (5,)
        self._register_param('bias', np.arange(5.0))

    def forward(self, x):
        # simple, simply passes the input around.
        return x

    def backward(self, g):
        # updates the gradient all by +1 (this is just a demo)
        self.grads['weight'] += 1
        self.grads['bias'] += 1
        # returns g, since this is a lambda function.
        return g


class MyNN(Module):
    def __init__(self):
        super().__init__()
        # a weight parameter of shape (2,2)
        self._register_param('weight', np.arange(4.0).reshape(2,2))
        # a bias parameter of shape (2,)
        self._register_param('bias', np.arange(2.0))
        self._register_child('inner', MyNNLambda())

    def forward(self, x):
        # call the child's forward
        return self.children['inner'].forward(x)

    def backward(self, g):
        self.grads['weight'] += 2
        self.grads['bias'] += 2
        # call the child's backward()
        return self.children['inner'].backward(g)


def demo_module_basic():
    my_module = MyNN()
    # iterate over all parameters and grads
    for name, param, grad in my_module.named_parameters():
        print(name, param.shape)
        assert param.shape == grad.shape
        # print gradient
        print(grad)
    print('\n'*10)
    # input a 2x3 input
    input_x = np.arange(6.0).reshape(2,3)
    print('input_x', input_x)
    # call the network.
    output_x = my_module.forward(input_x)
    # should be a same, as the module does nothing.
    print('output_x', output_x)

    # do back prop.
    # here we just put in some dummy grad.
    my_module.backward(np.arange(6.0).reshape(2,3))

    # gradients should be updated now. inner parameters got added by 1,
    # outer ones got added by 2.
    print('\n' * 10)
    for name, param, grad in my_module.named_parameters():
        print(name, param.shape)
        assert param.shape == grad.shape
        # print gradient
        print(grad)


if __name__ == '__main__':
    demo_module_basic()
