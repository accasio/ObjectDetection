import utils
import numpy as np
import pandas as pd


if __name__ == '__main__':
    split_val = 20
    file = utils.read_file()
    translation = utils.read_translation()

    # utils.train_test_ordered(file['Class ID'].get_values(), train['Class ID'].get_values(), test['Class ID'].get_values())
    # utils.class_distribution_ordered(train['Class ID'].get_values())
    # train.to_csv('train.csv', sep=',', index=False)
    # file = utils.move_col_to_back(file, 1)
    # file = utils.convert_ids_to_names(file, translation)
    # file.to_csv(utils.dir + 'test.txt', sep=',', index=False)