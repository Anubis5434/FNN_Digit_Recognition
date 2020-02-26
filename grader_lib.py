# check that my JSON is well formed.
# typically, I check `r == loads(dumps(r))` before returing `r`,
# making sure r can be fully dumped (rather than partially dumped due to
# floating number precision or some other issues).
from os.path import abspath, join, dirname
from json import dumps, loads
import numpy as np

root_dir = abspath(dirname(__file__))

question_json_mapping = {
    'q1': 'q1_out.json',
    'q2': 'q2_out.json',
    'q3': 'q3_out.json',
    'q4': 'q4_out.json',        
}

num_case_mapping = {
    'student': 1,
    'autolab': 10,
}

num_case_q4_mapping = {
    'student': 1,
    'autolab': 3,
}

grad_data_mapping = {
    'student': join(root_dir, 'layer_data_student'),
    'autolab': join(root_dir, 'layer_data_autolab'),
}

training_data_mapping = {
    'student': join(root_dir, 'training_data_student'),
    'autolab': join(root_dir, 'training_data_autolab'),
}


def verifiy_json_training_questions(result_: dict, num_epoch: int,
                                    train_size: int, test_size: int):
    # for q4

    # TODO: in the actual grader part 2,
    #   check that MD5 of grader_lib is not tempered.
    #   in the Makefile,
    #   check that MD5 of grader2 is not tempered.
    #   between part1 and part2, recheck all files (with &>/dev/null)
    #   making sure jsons are correct.

    result = loads(dumps(result_))
    assert loads(dumps(result_)) == loads(dumps(result))
    assert result == result_
    assert type(result) is dict
    assert result.keys() == {
        'train_loss',
        'test_loss',
        'train_error',
        'test_error',
        'yhat_train',
        'yhat_test',
    }

    for key in result:
        v = result[key]
        assert type(v) is list
        if key in {'train_loss', 'test_loss', 'train_error', 'test_error'}:
            assert len(v) == num_epoch
            for x in v:
                assert type(x) is float
        elif key == 'yhat_train':
            assert len(v) == train_size
            for x in v:
                assert type(x) is int
        elif key == 'yhat_test':
            assert len(v) == test_size
            for x in v:
                assert type(x) is int
        else:
            raise ValueError

    return result


def verifiy_json_grad_questions(result: dict, *, test_type,
                                **kwargs):
    # for q1 to q3

    assert type(result) is dict

    if test_type == 'linear':
        in_features = kwargs['in_features']
        out_features = kwargs['out_features']
        batch_size = kwargs['batch_size']

        shape_dict = {
            'output': (batch_size, out_features),
            'grad_input': (batch_size, in_features),
            'grad_weight': (out_features, in_features),
            'grad_bias': (out_features,),
        }

    elif test_type == 'sigmoid':
        in_features = kwargs['in_features']
        batch_size = kwargs['batch_size']

        shape_dict = {
            'output': (batch_size, in_features),
            'grad_input': (batch_size, in_features),
        }

    elif test_type == 'crossentropy':
        num_class = kwargs['num_class']
        batch_size = kwargs['batch_size']

        shape_dict = {
            'output': (),
            'grad_input': (batch_size, num_class),
        }
    else:
        raise ValueError

    # check keys
    result_json = {}
    assert result.keys() == shape_dict.keys()
    for k, s in shape_dict.items():
        v = np.asarray(result[k])
        assert v.shape == s
        assert v.dtype == np.float64
        result_json[k] = v.tolist()

    result_json = loads(dumps(result_json))
    assert result_json == loads(dumps(result_json))

    return result_json


def load_data(mode):
    data_train = np.genfromtxt(join(training_data_mapping[mode], 'train.csv'), 
        dtype=np.float64, delimiter=',')
    if mode == 'student':
        assert data_train.shape == (3000, 785)
    data_test = np.genfromtxt(join(training_data_mapping[mode], 'test.csv'),
        dtype=np.float64, delimiter=',')
    if mode == 'student':
        assert data_test.shape == (1000, 785)

    return {
        'train_x': data_train[:, :-1].astype(np.float64),
        'train_y': data_train[:, -1].astype(np.int64),
        'test_x': data_test[:, :-1].astype(np.float64),
        'test_y': data_test[:, -1].astype(np.int64),
    }


