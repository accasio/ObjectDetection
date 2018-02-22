import utils
import numpy as np
import pandas as pd


if __name__ == '__main__':
    split_val = 20
    file = utils.read_file()
    translation = utils.read_translation()
    # utils.class_distribution_ordered(file['Class ID'].get_values())
    # utils.class_distribution_ordered(file['Class ID'].get_values())
    # file.to_csv('filtered_GT.txt', sep=';', index=False)

    file = utils.filter_file_by_class_count(file)
    train, test = utils.create_train_test(file)

    # utils.train_test_ordered(file['Class ID'].get_values(), train['Class ID'].get_values(), test['Class ID'].get_values())
    # utils.class_distribution_ordered(train['Class ID'].get_values())
    # utils.class_distribution_ordered(test['Class ID'].get_values())
    train.to_csv('train.csv', sep=',', index=False)
    test.to_csv('test.csv', sep=',', index=False)
