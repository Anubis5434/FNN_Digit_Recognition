from sys import argv
from os.path import abspath, join, dirname
from json import dumps, load

import numpy as np

from grader_lib import (question_json_mapping, verifiy_json_grad_questions,
                        verifiy_json_training_questions, load_data,
                        num_case_mapping, grad_data_mapping,
                        training_data_mapping,
                        num_case_q4_mapping
                        )
from hw4_sol import (compute_linear,
                     compute_sigmoid,
                     compute_crossentropy,
                     autolab_trainer
                     )

root_dir = abspath(dirname(__file__))


def get_result_q1_to_q3(q, mode):
    num_case = num_case_mapping[mode]
    data_dir = grad_data_mapping[mode]

    func = {
        'q1': compute_linear,
        'q2': compute_sigmoid,
        'q3': compute_crossentropy,
    }[q]

    result_all = []

    for idx_case in range(num_case):
        # read data.
        with open(join(data_dir, q + '_' + str(idx_case) + '_in.json'), 'r',
                  encoding='utf-8') as f_data:
            data_this = load(f_data)

        data_this_input = data_this['input']
        for k in data_this_input:
            data_this_input[k] = np.asarray(data_this_input[k])
        result_this = func(**data_this_input)
        result_this = verifiy_json_grad_questions(result_this,
                                                  **data_this['verify'])
        result_all.append(result_this)

    return result_all


def get_result_q4(q, mode):
    num_case = num_case_q4_mapping[mode]
    data_dir = training_data_mapping[mode]
    dataset = load_data(mode)

    result_all = []

    for idx_case in range(num_case):
        with open(join(data_dir, q + '_' + str(idx_case) + '_in.json'), 'r',
                  encoding='utf-8') as f_data:
            data_this = load(f_data)

        data_this_input = data_this['input']
        for k in data_this_input:
            data_this_input[k] = np.asarray(data_this_input[k])

        result_this = autolab_trainer(dataset, **data_this_input)
        result_this = verifiy_json_training_questions(result_this,
                                                      **data_this['verify'])
        result_all.append(result_this)

    return result_all


def main():
    q, mode = argv[1:]
    assert q in question_json_mapping
    file_to_write = join(root_dir, question_json_mapping[q])
    if q in {'q1', 'q2', 'q3'}:
        # each one 10 cases.
        result = get_result_q1_to_q3(q, mode)
    elif q in {'q4'}:
        # 4 cases                
        result = get_result_q4(q, mode)
    else:
        raise ValueError

    with open(file_to_write, 'w', encoding='utf-8') as f:
        f.write(dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
