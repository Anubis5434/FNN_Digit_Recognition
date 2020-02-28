"""a minimal PyTorch like framework for NN"""

# NOTATION
# in the comments below
#     N is batch size.
#     in_features is the input size (not counting the bias)
#     out_features is the output size
#     C is number of classes (10 for this assignment)

import numpy as np


class Module:
    def __init__(self):
        super().__init__()
        self.params = dict()
        self.grads = dict()
        self.children = dict()
        self.cache = dict()

    def _register_param(self, name: str, param: np.ndarray):
        """ the parameter can be accessed via self.params[name]
        the gradient can be accessed via self.grads[name]
        """
        assert isinstance(param, np.ndarray)
        self.params[name] = param
        self.grads[name] = np.zeros_like(param)

    def _register_child(self, name: str, child: 'Module'):
        """ the module can be acccessed via self.children[name]"""
        assert isinstance(child, Module)
        self.children[name] = child

    def forward(self, *x):
        raise NotImplementedError

    def backward(self, *g):
        raise NotImplementedError

    def named_parameters(self, base: tuple = ()):
        """recursively get all params in a generator"""
        assert self.params.keys() == self.grads.keys()
        for name in self.params:
            full_name = '.'.join(base + (name,))
            yield full_name, self.params[name], self.grads[name]

        # recursively on others
        for child_name, child in self.children.items():
            yield from child.named_parameters(base=base + (child_name,))

    def clear_cache(self):
        """this is really not necessary in this assignment, just
        for completeness.
        """
        self.cache.clear()
        for child in self.children.values():
            child.clear_cache()


def sigmoid(x):
    """
    :param x: np.ndarray
    :return: np.ndarray, same shape as x, elementwise sigmoid of x
    """
    return 1 / (1 + np.exp(-x))


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        :param x: np.ndarray, shape (N, in_features)
        :return: np.ndarry, elementwise sigmoid of input, same shape as x.

        in terms of the writeup,

        this layer computes \vec{z} given \vec{a}.
        """
        self.cache['sigma'] = sigmoid(x)
        return sigmoid(x)

    def backward(self, g):
        """
        :param g: np.ndarray, shape (N, in_features), the gradient of
               loss w.r.t. output of this layer.
        :return: np.ndarray, shape (N, in_features), the gradient of
                 loss w.r.t. input of this layer.
        """
        sigma = self.cache['sigma']
        g_input = g * sigma * (1 - sigma)
        return g_input


class Linear(Module):
    def __init__(self, weight, bias):
        super().__init__()
        # weight has shape (out_features, in_features)
        self._register_param('weight', weight)
        # bias has shape (out_features,)
        self._register_param('bias', bias)

    def forward(self, x):
        """input has shape (N, in_features)
        and output has shape (N, out_features)

        in terms of the writeup,

        this layer computes \vec{a} given \vec{x},
        or \vec{b} given \vec{z}.
        """
        x_out = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)  # shape(N, in + 1)
        self.cache['x_out'] = x_out
        bias_reshape = self.params['bias'].reshape((self.params['bias'].shape[0], 1))  # shape(out, 1)
        weight_out = np.concatenate((bias_reshape, self.params['weight']), axis=1)  # shape(out, in + 1)
        self.cache['weight_out'] = weight_out
        output = weight_out @ x_out.T
        return output.T  # shape(N, out)

    def backward(self, g):
        """g is of shape (N, out_features)
        g_input should be of shape (N, in_features)"""
        updater = g.T @ self.cache["x_out"]  # shape(out, in + 1)
        self.grads['weight'] += updater[:, 1:]  # shape(out, in)
        self.grads['bias'] += updater[:, 0]  # shape(out,)
        g_input_origin = g @ self.cache['weight_out']  # shape(N, in + 1)
        g_input = g_input_origin[:, 1:]  # shape(N, in)
        return g_input


class CrossEntropyLoss(Module):
    """softmax + cross entropy loss"""

    def __init__(self):
        super().__init__()

    def forward(self, score, label):
        """
        :param score: np.ndarray, shape (N, C), the score values for
               input of softmax ($b_k$ in the writeup).
        :param label: integer-valued np.ndarray, shape (N,), all in [0,C-1]
               (non-zero idx of $\vec{y}$ in the writeup).
        :return: the mean negative cross entropy loss
               ($J(\alpha, \beta)$ in the write up).
        """
        N = score.shape[0]
        self.cache['N'] = N
        label = label.reshape((N, 1))  # shape(N, 1)
        trans_label = np.zeros((N, score.shape[1]))  # shape(N, C)
        for i in range(N):
            flag = label[i][0]
            trans_label[i, flag] = 1
        self.cache['label'] = trans_label
        exps = np.exp(score - np.max(score))  # shape(N, C)
        y_Hat = exps / np.sum(exps, axis=1).reshape((-1, 1))  # shape(N, C)
        self.cache['y_Hat'] = y_Hat
        cross_entropy_loss = (-1 / N) * np.sum(np.log(y_Hat) * trans_label)
        self.cache['loss'] = cross_entropy_loss
        self.cache['label_predict'] = np.argmax(y_Hat, axis=1)
        return cross_entropy_loss

    def backward(self):
        """returns the gradient of loss w.r.t. `score`"""
        return (-1 / self.cache['N']) * (self.cache['label'] - self.cache['y_Hat'])
