import tensorflow as tf

def load_cifar10_subset(grayscale=True):
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    train_images = train_images[:5000].astype("float32") / 255.0
    test_images = test_images[:1000].astype("float32") / 255.0

    train_labels = train_labels[:5000].reshape(-1)
    test_labels = test_labels[:1000].reshape(-1)

    train_images = tf.image.resize(train_images, [32, 32])
    test_images = tf.image.resize(test_images, [32, 32])

    if grayscale:
        train_images = tf.image.rgb_to_grayscale(train_images)
        test_images = tf.image.rgb_to_grayscale(test_images)

    class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
                   'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

    return train_images, train_labels, test_images, test_labels, class_names