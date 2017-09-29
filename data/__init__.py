import os
import random
from data.image import Image
import numpy
from shutil import copy, rmtree


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
    :return: A 3-tuple (train_data, validation_data, test_data). Each element of the 3-tuple is a 2-tuple. This is a 
        pair of (image_paths, labels)
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

    train_data, train_labels = [], []
    validation_data, validation_labels = [], []
    test_data, test_labels = [], []
    for item in os.listdir(path):
        if os.path.isfile(os.path.join(path, item)):
            continue

        filelist = [os.path.join(path, item, file) for file in os.listdir(os.path.join(path, item))
                    if os.path.isfile(os.path.join(path, item, file))]
        random.shuffle(filelist)
        size = len(filelist)

        temp = filelist[:int(size * train)]
        train_data.extend(temp)
        train_labels.extend([item] * len(temp))

        temp = filelist[int(size * train):int(size * (train + validation))]
        validation_data.extend(temp)
        validation_labels.extend([item] * len(temp))

        temp = filelist[int(size * (train + validation)):]
        test_data.extend(temp)
        test_labels.extend([item] * len(temp))

    return (train_data, train_labels), (validation_data, validation_labels), (test_data, test_labels)


def generate_and_save_features(img_paths, feature_func=Image.HoG, save_file=None, force_generate=False):
    """
    Generate image features and save to csv
    :param feature_func: Feature function from the Image class
    :param img_paths: List of paths to images
    :param save_file: File to save the features in
    :param force_generate: Generate the image features even if a saved file already exists
    :return: 
    """
    if not force_generate and os.path.exists(save_file):
        return numpy.genfromtxt(save_file, delimiter=',', defaultfmt="%d")

    features = [feature_func(Image(x, as_grey=True)) for x in img_paths]
    if save_file:
        numpy.savetxt(save_file, features, fmt="%d", delimiter=',')
    return features


def generate_toy_dataset(origin, destination, num=50, force_generate=False):
    if os.path.exists(destination) and not force_generate:
        return

    if os.path.exists(destination):
        rmtree(destination)

    os.mkdir(destination)
    for item in os.listdir(origin):
        if os.path.isfile(os.path.join(origin, item)):
            continue

        os.mkdir(os.path.join(destination, item))

        filelist = [os.path.join(origin, item, file) for file in os.listdir(os.path.join(origin, item))
                    if os.path.isfile(os.path.join(origin, item, file))]
        random.shuffle(filelist)
        size = len(filelist)

        for i in range(min(size, num)):
            copy(os.path.join(origin, item, filelist[i]), os.path.join(destination, item))


def accuracy(values, expected):
    if len(values) != len(expected):
        raise ValueError("Sizes of two lists must be same")
    return numpy.sum(numpy.array(values) == numpy.array(expected)) / len(values)
