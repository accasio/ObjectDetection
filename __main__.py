import utils
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    file = utils.read_file()

    # regulatory_mand = file.loc[file['Class ID'].isin(range(2, 6))]
    # utils.class_distribution(file['Class ID'].get_values())
    utils.ordered_dist(file['Class ID'].get_values())
    # train_addrs, train_labels, test_addrs, test_labels = utils.train_test(regulatory_mand, shuffle_data=False)


    # file['height'] = height
    # file['width'] = width

    # file.to_csv('update GT.csv', sep=';', encoding='utf-8')
