import tensorflow as tf
from tensorflow.keras import layers, models

class ModelDeepDenseNet:
    def __init__(self, train_images, train_labels, test_images, test_labels, class_names, image_size_x, image_size_y, image_channels):
        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels
        self.class_names = class_names
        self.classifier_name = "DenseNet"
        self.input_shape = image_size_x * image_size_y  
        self.build_model()

    def build_model(self):
        self.model = models.Sequential([
            tf.keras.Input(shape=(self.input_shape,)),  
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(len(self.class_names), activation='softmax')
        ])

        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def prepare_data(self):
        
        self.x_train = self.train_images.numpy().reshape((len(self.train_images), -1))
        self.x_test = self.test_images.numpy().reshape((len(self.test_images), -1))

    def augment_data(self):
        
        pass

    def fit(self):
        self.prepare_data()  
        self.model.fit(self.x_train, self.train_labels, epochs=10, verbose=0)

    def predict(self):
        self.pred_labels = tf.argmax(self.model.predict(self.x_test), axis=1).numpy()

    def evaluate(self):
        self.model.evaluate(self.x_test, self.test_labels, verbose=2)


def build_model(train_images, train_labels, test_images, test_labels, class_names, image_size_x=32, image_size_y=32, image_channels=1):
    model = ModelDeepDenseNet(train_images, train_labels, test_images, test_labels, class_names, image_size_x, image_size_y, image_channels)
    return model
