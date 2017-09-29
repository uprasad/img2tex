import skimage.io as image_io
import skimage.feature as image_feature


class Image:
    """
    Collection of methods to extract features from raw images
    """

    def __init__(self, path, as_grey=False):
        """
        Reads the image at path
        :param path: Absolute path of the image
        :param flags: OpenCV flags to be passed to cv2.imread()
        """
        self.img = image_io.imread(path, as_grey)

    def raw(self):
        """
        Returns the raw image pixels as a feature vector
        :return: 1-D vector of image pixels
        """
        return self.img.flatten()

    def HoG(self):
        """
        Calculate and returns the Histogram of Gradient descriptor for the image
        :return: Histogram of Gradient features for the image 
        """
        # these parameters are for a 45x45 image
        features = image_feature.hog(
            self.img,
            orientations=9,
            pixels_per_cell=(5, 5),
            cells_per_block=(2, 2),
            visualise=False,
            feature_vector=True,
            block_norm="L2-Hys"
        )
        return features
