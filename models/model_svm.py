from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

class ModelSVM:
    def __init__(self, train_images, train_labels, test_images, test_labels, class_names, image_size_x, image_size_y, image_channels):
        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels
        self.class_names = class_names
        self.classifier_name = "SGD"
        self.scaler = StandardScaler()
        self.model = SGDClassifier(max_iter=1000, tol=1e-3)

    def prepare_data(self):
        x_train = self.train_images.numpy().reshape((len(self.train_images), -1))
        x_test = self.test_images.numpy().reshape((len(self.test_images), -1))
        self.x_train = self.scaler.fit_transform(x_train)
        self.x_test = self.scaler.transform(x_test)

    def augment_data(self):
        original_labels = np.ravel(self.train_labels.copy())  
        repeat_factor = 5
        self.x_train = np.tile(self.x_train, (repeat_factor, 1))        # 5000 → 25000
        self.train_labels = np.tile(original_labels, repeat_factor)    # 5000 → 25000
        assert len(self.x_train) == len(self.train_labels), f"Still inconsistent: {len(self.x_train)} vs {len(self.train_labels)}"

    def fit(self):
        print(f"x_train shape: {self.x_train.shape}")
        print(f"train_labels shape: {self.train_labels.shape}")
        self.model.fit(self.x_train, self.train_labels)
    def predict(self):
        self.pred_labels = self.model.predict(self.x_test)

    def evaluate(self):
        from sklearn.metrics import classification_report
        print(classification_report(self.test_labels, self.pred_labels, target_names=self.class_names))


def build_model(train_images, train_labels, test_images, test_labels, class_names, image_size_x=32, image_size_y=32, image_channels=3):
    model = ModelSVM(train_images, train_labels.copy(), test_images, test_labels.copy(), class_names, image_size_x, image_size_y, image_channels)
    model.prepare_data()
    model.augment_data()
    return model
