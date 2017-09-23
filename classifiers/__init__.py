import abc


class Classifier(object):
    """
    Abstract class to outline the functionality of a classifier
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def train(self, data, labels):
        """
        Method for training the classifier
        :param data: The features of the training data
        :param labels: The labels of the training data
        :return: Model learned
        """

    @abc.abstractmethod
    def predict(self, model, data):
        """
        Method to run a model against some data
        :param model: The model learned after training
        :param data: The data the model is run against
        :return: The predicted labels
        """
