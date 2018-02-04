import utils
import random
random.seed(1)
import pandas as pd

split_val = 20

if __name__ == '__main__':
    file = utils.read_file()
    translation = utils.read_translation()
    # utils.class_distribution_ordered(file['Class ID'].get_values())
    file = utils.filter_file_by_class_count(file)
    # utils.class_distribution_ordered(file['Class ID'].get_values())
    # file.to_csv('filtered_GT.txt', sep=';', index=False)
    groups = [df for _, df in file.groupby('filename')]
    random.shuffle(groups)
    num_groups = len(groups)
    split = num_groups * 0.8
    train = groups[:num_groups]
    test = groups[num_groups:]
    pd.concat(train).reset_index(drop=True)
    pd.concat(test).reset_index(drop=True)
    print(test)

