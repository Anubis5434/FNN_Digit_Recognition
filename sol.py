import numpy as np
from lib import *
import matplotlib.pyplot as plt

# NOTATION
# in the comments below
#     N is batch size.
#     in_features is the input size (not counting the bias)
#     out_features is the output size
#     C is number of classes (10 for this assignment)
#     E is the number of epochs


def compute_linear(x: np.ndarray,
                   weight: np.ndarray,
                   bias: np.ndarray,
                   grad_higher: np.ndarray):
    # given input (x), weight, bias, and gradient of output,
    # compute the output value, and gradients w.r.t. input (x), weight,
    # and bias.
    # save your result as a dictionary, with keys
    # `output`, `grad_input`, `grad_weight`, `grad_bias`,
    """
    :param x: np.ndarray, shape (N, in_features), input of the linear layer
    :param weight: np.ndarray, shape (output_feature, in_feature), the
                    weight matrix of the linear layer
    :param bias: np.ndarray, shape (output_feature,), the bias term added
                to the output
    :param grad_higher: np.ndarray, shape (N, out_features), the gradient
                of loss w.r.t. output of this layer

    :return: should be a dictionary containing following keys:

         'output':  np.ndarray, shape (N, out_features), the output of the
                    linear layer

         'grad_input':  np.ndarray, shape (N, in_features), the gradient
                        of loss w.r.t. input of this layer

         'grad_weight':  np.ndarray, shape (out_features, in_features),
                         the gradient of loss w.r.t. weight matrix of
                         this layer

         'grad_bias':  np.ndarray, shape (out_features,), the gradient
                       of loss w.r.t. bias of this layer
    """
    result = {}
    linear1 = Linear(weight, bias)
    result["output"] = linear1.forward(x)
    result['grad_input'] = linear1.backward(grad_higher)
    result['grad_weight'] = linear1.grads["weight"]
    result['grad_bias'] = linear1.grads['bias']
    return result


def compute_sigmoid(x: np.ndarray, grad_higher: np.ndarray):
    # given input (x), and gradient of output,
    # compute the output value, and gradients w.r.t. input (x).
    # save your result as a dictionary, with keys
    # `output`, `grad_input`
    """
    :param x: np.ndarray, shape (N, in_features), input of the sigmoid layer
    :param grad_higher: np.ndarray, shape (N, in_features), the gradient
                        of loss w.r.t. the output of this layer

    :return: should be a dictionary containing following keys:

        'output':  np.ndarray, shape (N, in_features), output of the
                   sigmoid layer

        'grad_input':  np.ndarray, shape (N, in_features), the gradient
                       of loss w.r.t. the input of this layer

    """
    result = {}
    sig = Sigmoid()
    result['output'] = sig.forward(x)
    result['grad_input'] = sig.backward(grad_higher)
    return result


def compute_crossentropy(score: np.ndarray,
                         label: np.ndarray):
    # given inputs (score and label),
    # compute the output value (loss), and gradients w.r.t. score.
    # save your result as a dictionary, with keys
    # `output`, `grad_input`
    """
    :param score: np.ndarray, shape (N, C), the score values for input of
                input of softmax ($b_k$ in the writeup).
                notice that here C can be other than 10.
    :param label: np.ndarray of integers, shape (N,), each value in [0,C-1]
                (non-zero idx of $\vec{y}$ in the writeup).

    :return: should be a dictionary containing following keys:

        'output': int, the mean negative cross entropy loss

        'grad_input': np.ndarray, shape (N, C), the gradient of loss
                      w.r.t. the score
    """
    result = {}
    c_e = CrossEntropyLoss()
    result["output"] = c_e.forward(score, label)
    result['grad_input'] = c_e.backward()
    return result


class HW4NN(Module):
    def __init__(self,
                 weight_l1, bias_l1,
                 weight_l2, bias_l2,
                 ):
        super().__init__()
        # register the parameters

        # initialize and register the child module
        linear_l1 = Linear(weight_l1, bias_l1)
        self._register_child('linear_l1', linear_l1)

        sigmoid1 = Sigmoid()
        self._register_child('sigmoid', sigmoid1)

        linear_l2 = Linear(weight_l2, bias_l2)
        self._register_child('linear_l2', linear_l2)

        C_E_L = CrossEntropyLoss()
        self._register_child('C_E_L', C_E_L)

    def forward(self, x, y):
        """
        :param x: np.ndarray, shape (N, in_features), training input
        :param y: np.ndarray, shape (N, ), training label

        :return: the mean negative cross entropy loss
        """
        # print(x)
        a = self.children['linear_l1'].forward(x)
        # print('pre sig', a)
        z = self.children['sigmoid'].forward(a)
        # print('post sig', z)
        b = self.children['linear_l2'].forward(z)
        c_e_l = self.children['C_E_L'].forward(b, y)
        return c_e_l

    def backward(self):
        partial_cel_bk = self.children['C_E_L'].backward()
        partial_cel_z = self.children['linear_l2'].backward(partial_cel_bk)
        partial_cel_beta = self.children['linear_l2'].params['weight']
        partial_cel_a = self.children['sigmoid'].backward(partial_cel_z)
        partial_cel_alpha = self.children['linear_l1'].backward(partial_cel_a)
        return partial_cel_alpha, partial_cel_beta


def define_network(init_scheme,
                   num_hidden,
                   seed=0, size_input=784,
                   size_output=10
                   ):
    """
    :param init_scheme: can be 'random' or 'zero', used to initialize the
                        weight matrices
    :param num_hidden: the number of hidden units
    :param seed: seed used to generate initial random weights. can be ignored.
    :param size_input: number of input features. can be ignored (784 for this
           assignment)
    :param size_output: number of output classes. can be ignored (10 for this
           assignment)
    """
    rng_state = np.random.RandomState(seed=seed)

    # generate weights according to init_scheme
    if init_scheme == 'random':
        weight_l1 = rng_state.randn(num_hidden, size_input)
        weight_l2 = rng_state.randn(size_output, num_hidden)
    if init_scheme == 'zero':
        weight_l1 = np.zeros((num_hidden, size_input))
        weight_l2 = np.zeros((size_output, num_hidden))

    # generate bias
    bias_l1 = np.zeros((num_hidden,))
    bias_l2 = np.zeros((size_output,))

    # generate network
    nn = HW4NN(weight_l1, bias_l1, weight_l2, bias_l2)
    return nn


def clear_grad(nn):
    # clear grad info in the parameters.
    # hint: use nn.named_parameters, and the fact that given
    # an array x, x[...] = 0 empties it in place.
    """
    clear the gradient in the parameters and replace them with 0's
    """
    for name, param, grad in nn.named_parameters():
        grad[...] = 0


def update_param(nn, lr):
    # update parameters
    # hint: use nn.named_parameters, and the fact that given
    # two arrays x, y of same shape,
    # x += y updates x to be x + y in place.
    """
    update the parameters of the network
    """
    for name, param, grad in nn.named_parameters():
        param -= lr * grad


def train_network(nn, dataset, num_epoch,
                  learning_rate,
                  seed=0, no_shuffle=True, ):
    """
    :param nn: neural network object
    :param dataset: a dictionary of following keys:
            'train_x': np.ndarray, shape (N, 784)
            'train_y': np.ndarray, shape (N, )
            'test_x': np.ndarray of int, shape (N_test, 784)
            'test_y': np.ndarray of int, shape (N_test, )

            for training_data_student, we should have
            N=3000, and N_test=1000

    :param num_epoch: (E) the number of epochs to train the network
    :param learning_rate: the learning_rate multiplied on the gradients
    :param seed: an integer used to generate random initial weights,
           not needed for autolab.
    :param no_shuffle: a boolean indicating if the training dataset will be
                shuffled before training,
           keep default value for autolab.

    :return: should be a dictionary containing following keys:

        'train_loss': list of training losses, its size should equal E


        'test_error': list of testing losses, its size should equal E

        'train_error': list of training errors, its size should equal E

        'test_error': list of testing errors, its size should equal E

        'yhat_train': list of prediction labels for training dataset,
                      its size should equal N

        'yhat_test': list of prediction labels for testing dataset,
                     its size should equal N_test
    """
    # get data
    train_x, train_y = dataset['train_x'], dataset['train_y']
    test_x, test_y = dataset['test_x'], dataset['test_y']

    rng_state = np.random.RandomState(seed=seed)

    training_loss_all = []
    training_error_all = []
    testing_loss_all = []
    testing_error_all = []

    for idx_epoch in range(num_epoch):  # for each epoch
        if no_shuffle:
            # this is for autolab.
            shulffe_idx = np.arange(train_y.size)
        else:
            # this is for empirical questions.
            shulffe_idx = rng_state.permutation(train_y.size)
        for idx_example in shulffe_idx:  # for each training sample.
            x_this = train_x[idx_example:idx_example + 1]
            y_this = train_y[idx_example:idx_example + 1]

            # clear grad
            clear_grad(nn)

            # forward
            nn.forward(x_this, y_this)

            # generate grad
            nn.backward()

            # update parameters
            update_param(nn, learning_rate)

        # now we arrive at the end of this epoch,
        # we want to compute some statistics.

        # training_loss_this_epoch is average loss over training data.
        # note that you should compute this loss ONLY using the model
        # obtained at the END of this epoch,
        # i.e. you should NOT compute this loss ON THE FLY by averaging
        # intermediate training losses DURING this epoch.
        nn.forward(train_x, train_y)
        training_loss_this_epoch = nn.children["C_E_L"].cache['loss']

        # record training loss
        # this float() is just there so that result can be JSONified easily
        training_loss_all.append(float(training_loss_this_epoch))

        # generate predicted labels for training data
        # yhat_train_all should be a 1d vector of same shape as train_y
        yhat_train_all = nn.children['C_E_L'].cache['label_predict']

        # record training error
        training_error_all.append(float((yhat_train_all != train_y).mean()))

        # training_loss_this_epoch is average loss over training data.
        nn.forward(test_x, test_y)
        testing_loss_this_epoch = nn.children["C_E_L"].cache['loss']

        # record testing loss
        testing_loss_all.append(float(testing_loss_this_epoch))

        # generate yhat for testing data
        yhat_test_all = nn.children['C_E_L'].cache['label_predict']

        # record testing error
        testing_error_all.append(float((yhat_test_all != test_y).mean()))

    # keep this part intact, do not modify it.
    return {
        # losses and errors across epochs.
        'train_loss': training_loss_all,
        'test_loss': testing_loss_all,
        'train_error': training_error_all,
        'test_error': testing_error_all,
        # yhat of the final model at the last epoch.
        # tolist for JSON.
        'yhat_train': yhat_train_all.tolist(),
        'yhat_test': yhat_test_all.tolist(),
    }


def autolab_trainer(dataset, init_scheme, num_hidden,
                    num_epoch, learning_rate, size_input=784):
    """

    :param dataset: dataset as in that of `train_network`
    :param init_scheme: init_scheme as in that of `define_network`
    :param num_hidden: num_hidden as in that of `define_network`
    :param num_epoch: num_epoch as in that of `train_network`
    :param learning_rate: num_epoch as in that of `train_network`
    :param size_input: size_input as in that of `define_network`,
           can be ignored fir this assignment
    :return: return value of `train_network`.
    """
    nn = define_network(init_scheme, num_hidden)
    return train_network(nn, dataset, num_epoch, learning_rate)

# how to test your solution locally.

# nn = define_network(...)
# dataset = load_data(...)
# result = train_network(nn, dataset, ...)

# how to generate dataset ready for `train_network`
# when you are testing your data.


def load_data():
    data_train = np.genfromtxt('train.csv', dtype=np.float64, delimiter=',')
    assert data_train.shape == (3000, 785)
    data_test = np.genfromtxt('test.csv', dtype=np.float64, delimiter=',')

    assert data_test.shape == (1000, 785)

    return {
        'train_x': data_train[:, :-1].astype(np.float64),
        'train_y': data_train[:, -1].astype(np.int64),
        'test_x': data_test[:, :-1].astype(np.float64),
        'test_y': data_test[:, -1].astype(np.int64),
    }


if __name__ == "__main__":
    NN = define_network('random', num_hidden=100)
    num_epoch = 200
    dataset = load_data()
    train_result = train_network(NN, dataset, num_epoch, 0.01)
    # result
    print(train_result['train_error'], '\n', train_result['test_error'])
    plt.plot(x, train_result['train_error'], label='trainError')
    plt.plot(x, train_result['test_error'], label='testError')
    plt.legend()
    plt.xlabel('num_epoch')
    plt.show()
