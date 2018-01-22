import utils
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    file = utils.read_file()
    # translation = utils.read_translation()
    utils.class_distribution_ordered(file['Class ID'].get_values())


    # sign_classes, class_indices, class_counts = np.unique(file['Class ID'].get_values(), return_index=True, return_counts=True)
    # class_counts, sign_classes = zip(*sorted(zip(class_counts, translation['class_name'].get_values())))
    #
    # np.savetxt("class_counts.csv", np.column_stack((sign_classes, class_counts)), delimiter=",", fmt='%s')

    # file['height'] = height
    # file['width'] = width

    # file.to_csv('update GT.csv', sep=';', encoding='utf-8')
