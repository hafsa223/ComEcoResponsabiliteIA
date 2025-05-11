import tensorflow as tf 
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16

class ModelDeepVGGNet:
    def __init__(self, train_images, train_labels, test_images, test_labels, class_names, image_size_x, image_size_y, image_channels, pretrained=True):
        if train_images.shape[-1] == 1:
            self.train_images = tf.repeat(train_images, 3, axis=-1)
            self.test_images = tf.repeat(test_images, 3, axis=-1)
        else:
            self.train_images = train_images
            self.test_images = test_images

        self.train_labels = train_labels
        self.test_labels = test_labels
        self.class_names = class_names
        self.classifier_name = "VGG pretrained" if pretrained else "VGG untrained"
        self.input_shape = (image_size_x, image_size_y, 3)  # forcé à 3 canaux
        self.pretrained = pretrained
        self.build_model()

    def build_model(self):
        if self.pretrained:
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape)
            base_model.trainable = False
            self.model = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(128, activation='relu'),
                layers.Dense(len(self.class_names), activation='softmax')
            ])
        else:
            self.model = models.Sequential([
                tf.keras.Input(shape=self.input_shape),
                layers.Conv2D(32, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.Flatten(),
                layers.Dense(128, activation='relu'),
                layers.Dense(len(self.class_names), activation='softmax')
            ])

        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def prepare_data(self):
        pass

    def augment_data(self):
        pass

    def fit(self):
        self.model.fit(self.train_images, self.train_labels, epochs=10, verbose=0)

    def predict(self):
        self.pred_labels = tf.argmax(self.model.predict(self.test_images), axis=1).numpy()

    def evaluate(self):
        self.model.evaluate(self.test_images, self.test_labels, verbose=2)

def build_model(train_images, train_labels, test_images, test_labels, class_names, image_size_x=32, image_size_y=32, image_channels=1, pretrained=True):
    model = ModelDeepVGGNet(train_images, train_labels, test_images, test_labels, class_names, image_size_x, image_size_y, image_channels, pretrained)
    model.prepare_data()
    model.augment_data()
    return model
