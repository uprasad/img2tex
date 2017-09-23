import os
import random


def split(path, train, validation, test=None):
    """
    Splits the input data into training, validation and test datasets in the given proportions
    :param path: directory of the input dataset
        Within this directory, the input data is expected to be organized as follows:
        path/
            class1/
                img1_1.jpg
                img1_2.jpg
                ...
            class2/
                img2_1.jpg
                ...
    :param train: The fraction of the data that we want to use for training
    :param validation: The fraction of the data that we want to use for validation and hyper-parameter estimation 
    :param test: The fraction of the data that we want to use for final testing
        (Optional) if valid training and validation fractions are given, this will be calculated
    :return: A tuple (train_data, validation_data, test_data). Each element of the tuple is a dict. The dict maps the 
        class names to the list of paths to images that belong to that class
    """
    if not test:
        if train + validation > 1.0:
            raise ValueError("Fraction of training and validation datasets exceeds 1.0")
        test = 1.0 - (train + validation)
    elif train + validation + test != 1.0:
        raise ValueError("Sum of proportions of train, validation and test is not 1.0")

    if train < 0 or validation < 0 or test < 0:
        raise ValueError("Fraction of data must be between 0 and 1 (incl.)")

    if not path:
        raise ValueError("Path to input dataset is invalid")

    train_data = {}
    validation_data = {}
    test_data = {}
    for item in os.listdir(path):
        filelist = [file for file in os.listdir(os.path.join(path, item))
                    if os.path.isfile(os.path.join(path, item, file))]
        random.shuffle(filelist)
        size = len(filelist)
        train_data[item], validation_data[item], test_data[item] = \
            filelist[:int(size * train)], filelist[int(size * train):int(size * (train + validation))], \
            filelist[int(size * (train + validation)):]

    return train_data, validation_data, test_data
