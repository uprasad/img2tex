import cv2


class ImageFeatures:
    """
    Collection of methods to extract features from raw images
    """
    
    @staticmethod
    def raw(path):
        """
        Returns the raw image pixels as a feature vector
        :param path: Absolute path to the raw image
        :return: 1-D vector of image pixels
        """
        img = cv2.imread(path)
        return img.flatten()
