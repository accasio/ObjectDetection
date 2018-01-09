import utils
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    file = utils.read_file()

    regulatory_mand = file.loc[file['Class ID'].isin(range(2, 6))]
    # utils.class_distribution(regulatory_mand['Class ID'].get_values())

    # train_addrs, train_labels, test_addrs, test_labels = utils.train_test(regulatory_mand, shuffle_data=False)

    height = []
    width = []

    for i in range(file.shape[0]):
        if file['Image Source'][i] == 'Google Street View':
            width.append(1917)
            height.append(977)
        elif file['Image Source'][i] == 'Panasonic Camera':
            width.append(4592)
            height.append(3448)
        else:
            width.append(3840)
            height.append(2160)


    print(file['Image Source'].get_values())
    print(height)
    # file['height'] = height
    # file['width'] = width

    # file.to_csv('update GT.csv', sep=';', encoding='utf-8')
