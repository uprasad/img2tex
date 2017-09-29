from data import split, generate_and_save_features, generate_toy_dataset, Image, accuracy
from classifiers.k_nearest_neighbors import KNearestNeighbors
from scipy.spatial.distance import cityblock

generate_toy_dataset("/Users/udbhav/Code/handwritten/extracted_images",
                     "/Users/udbhav/Code/handwritten/toy_images", 50, False)

(train_imgs, train_labels), (validation_imgs, validation_labels), (test_imgs, test_labels) = \
    split("/Users/udbhav/Code/handwritten/toy_images", 0.9, 0.1)

train_feat_file = "data/train_feat.dat"
validation_feat_file = "data/validation_feat.dat"
test_feat_file = "data/test_feat.dat"

train_feat = generate_and_save_features(train_imgs, Image.HoG, train_feat_file, force_generate=True)
validation_feat = generate_and_save_features(validation_imgs, Image.HoG, validation_feat_file, force_generate=True)
test_feat = generate_and_save_features(test_imgs, Image.HoG, test_feat_file, force_generate=True)

for k in [1, 3, 5, 7, 9]:
    knn = KNearestNeighbors(k, dist=cityblock)
    model = knn.train(data=train_feat, labels=train_labels)
    predicted_labels = knn.predict(model=model, data=validation_feat)

    for i in range(len(predicted_labels)):
        print(predicted_labels[i], validation_labels[i])

    # print(k, accuracy(predicted_labels, validation_labels))
