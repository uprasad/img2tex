import classifiers
import heapq


def majority(arr, default=None):
    """
    
    :param arr: List of values from which we want to calculate majority
    :param default: 
    :return: 
    """
    candidate = default
    count = 0
    for elem in arr:
        if count != 0:
            count = count + (1 if candidate == elem else -1)
        else:
            count = 1
            candidate = elem

    return candidate if arr.count(candidate) > len(arr) // 2 else default


class KNearestNeighbors(classifiers.Classifier):
    def __init__(self, k, dist):
        self.k = k
        self.dist = dist

    def train(self, data, labels):
        """
        Simply memorize the training data and labels
        :param data: NxD matrix where N is the number of input images and D is the length of the feature vector for
            each image
        :param labels: Nx1 matrix containing the class of each training image
        :return: 
        """
        return data, labels

    def predict(self, model, data):
        """
        Compares each test image feature against the training images and calculates the 'k' closest images using the 
            'dist' distance metric
        :param model: The learned model, a 2-tuple (training_data, training_labels)
        :param data: Test input data NxD matrix where N is the number of test images, and D is the length of each 
            image's feature vector
        :return: 
        """
        if not model:
            raise ValueError("Model not defined")

        if data is None:
            raise ValueError("Data not defined")

        train_data, train_labels = model
        predicted_labels = []
        q = []
        for datum in data:
            for i in range(len(train_data)):
                # putting -1*dist in heap because we want to simulate a max-heap
                heapq.heappush(q, (-self.dist(datum, train_data[i]), train_labels[i]))
                if len(q) > self.k:
                    heapq.heappop(q)
            # predicted_labels.append(majority([e[1] for e in q]))
            predicted_labels.append([e[1] for e in q])

        return predicted_labels
