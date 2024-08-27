import json

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from matplotlib import cm

imitation = 'imitation'
classical = 'classical_log'


def get_reward_and_loss(log: str, max_len=-1, offset=0):
    with open(log, 'r') as log_file:
        logs = log_file.readlines()
        log_file.close()

    logs = logs[1 + offset:max_len]

    x = [float(x * 20) for x in range(1, len(logs) + 1)]


    reward = [float(x.split()[-6]) for x in logs]


    loss = [float(x.split()[-4]) for x in logs]

    q_value = [float(x.split()[-3]) for x in logs]

    x = np.array(x)

    reward = np.array(reward)

    loss = np.array(loss)

    q_value = np.array(q_value)

    return x, reward, loss, q_value


def draw_figure(imitation_x, imitation_data, classical_x, classical_data, title='default_title',
                figure_save_dir=None, first_data_label=classical, second_data_label=imitation):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)  # Create a figure containing a single Axes.
    # ax.plot(imitation_learning_x, imitation_learning_loss, label=imitation, color='r')
    fig.suptitle(title)
    ax1.plot(classical_x, classical_data, label=second_data_label, color='b')
    ax2.plot(imitation_x, imitation_data, label=first_data_label, color='r')
    ax3.plot(classical_x, classical_data, label=second_data_label, color='b')
    ax3.plot(imitation_x, imitation_data, label=first_data_label, color='r')
    ax1.legend()
    ax2.legend()
    ax3.legend()
    # ax2.legend()
    # Plot some data on the Axes.
    plt.show()
    if figure_save_dir:
        plt.savefig(fname=figure_save_dir)


def compare_multiple_log(log_datas, title, label, fig_title='default'):
    data_num = len(log_datas) - 1
    line_num = len(log_datas[0])
    x_datas = log_datas[0]
    y_datas = log_datas[1:]
    print(f'data_num {data_num}')
    fig, axs = plt.subplots(1, data_num, figsize=(20, 5))
    if data_num == 1:
        axs = [axs]
    for axs_index in range(0, data_num):
        for data_index in range(line_num):
            axs[axs_index].plot(x_datas[data_index], y_datas[axs_index][data_index], label=label[data_index])
        axs[axs_index].set_title(title[axs_index])
        axs[axs_index].legend()
    fig.suptitle(fig_title)
    plt.savefig(f'png of train result/{fig_title}')


def combine_logs(logs):
    log_item_num = len(logs[0])
    result = []
    for i in range(log_item_num):
        item_set = []
        for log in logs:
            item_set.append(log[i])
        result.append(item_set)
    return result


def compare_different_RL_log(log_list, fig_title='default'):
    log_datas = [get_reward_and_loss(x.get('log_url'), x.get('max_length'), x.get('offset')) for x in log_list]

    log_datas = combine_logs(log_datas)

    title = ['avg_reward', 'avg_loss', 'avg_q_value']
    label = [x.get('label') for x in log_list]

    compare_multiple_log(log_datas, title, label, fig_title=fig_title)

    return log_datas


imitation_complexity_cnn_32 = {"log_url": 'complexity_CNN_model_32/imitation_log',
                               "label": "complexity CNN model imitation",
                               "max_length": 501, "offset": 0}
classical_complexity_cnn_32 = {"log_url": 'complexity_CNN_model_32/classical_log',
                               "label": "complexity CNN model classical",
                               "max_length": 2001, "offset": 0}

continue_complexity_cnn_32 = {"log_url": 'complexity_CNN_model_32/classical_log',
                              "label": "complexity CNN model classical",
                              "max_length": 2001, "offset": 502}

imitation_default_cnn_32 = {"log_url": 'default_CNN_model_32:/imitation_log',
                            "label": "medium CNN model imitation",
                            "max_length": 501, "offset": 0}

classical_default_cnn_32 = {"log_url": 'default_CNN_model_32:/classical_log',
                            "label": "medium CNN model classical",
                            "max_length": 2001, "offset": 0}

continue_default_cnn_32 = {"log_url": 'default_CNN_model_32:/imitation_log',
                           "label": "medium CNN model continue",
                           "max_length": 2001, "offset": 502}

imitation_simplify_cnn_32 = {"log_url": 'simplify_CNN_model_64:/imitation_simplify',
                             "label": "simplify CNN model imitation",
                             "max_length": 501, "offset": 0}

classical_simplify_cnn_32 = {"log_url": 'simplify_CNN_model_64:/classical_simplify',
                             "label": "simplify CNN model classical",
                             "max_length": 2001, "offset": 0}

continue_simplify_cnn_32 = {"log_url": 'simplify_CNN_model_64:/classical_simplify',
                            "label": "simplify CNN model classical",
                            "max_length": 2001, "offset": 502}

classical_default_cnn_144 = {"log_url": 'default_cnn_144:/complexity_classical_log',
                             "label": "medium CNN model imitation 144",
                             "max_length": 2001, "offset": 0}

# vit model result

imitation_complexity_vit_32 = {"log_url": 'transformer_model_result/complexity_imitation_log',
                               "label": "complexity vit model imitation",
                               "max_length": 501, "offset": 0}
classical_complexity_vit_32 = {"log_url": 'transformer_model_result/complexity_classical_log',
                               "label": "complexity vit model classical",
                               "max_length": 2001, "offset": 0}

continue_complexity_vit_32 = {"log_url": 'transformer_model_result/complexity_imitation_log',
                              "label": "complexity vit model classical",
                              "max_length": 2001, "offset": 502}

imitation_default_vit_32 = {"log_url": 'transformer_model_result/default_imitation_log',
                            "label": "medium vit model imitation",
                            "max_length": 501, "offset": 0}

classical_default_vit_32 = {"log_url": 'transformer_model_result/default_classical_log',
                            "label": "medium vit model classical",
                            "max_length": 2001, "offset": 0}

continue_default_vit_32 = {"log_url": 'transformer_model_result/default_imitation_log',
                           "label": "medium vit model continue",
                           "max_length": 2001, "offset": 502}

# compare_different_RL_log([imitation_complexity_cnn_32, imitation_default_cnn_32, imitation_simplify_cnn_32],
#                          'compare cnn model imitation')
#
# compare_different_RL_log([classical_complexity_cnn_32, classical_default_cnn_32, classical_simplify_cnn_32],
#                          'compare cnn model classical')
#
# compare_different_RL_log([continue_default_cnn_32, classical_default_cnn_32],
#                          'compare continue medium CNN and classical CNN')
#
#
# compare_different_RL_log([imitation_complexity_vit_32, imitation_default_vit_32],
#                          'compare vit model imitation')
#
# compare_different_RL_log([classical_complexity_vit_32, classical_default_vit_32],
#                          'compare vit model classical')
#
# compare_different_RL_log([continue_default_vit_32, classical_default_vit_32],
#                          'compare continue medium vit and classical vit')
#
# compare_different_RL_log([imitation_default_vit_32, imitation_default_cnn_32],
#                          'compare cnn and vit model imitation')
#
# compare_different_RL_log([classical_default_vit_32, classical_default_cnn_32],
#                          'compare cnn and vit model classical')
