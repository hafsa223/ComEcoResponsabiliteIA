import time
from codecarbon import EmissionsTracker
from data_loader import load_cifar10_subset
from models.model_svm import build_model as build_svm
from models.model_dense import build_model as build_dense
from models.model_cnn import build_model as build_cnn
from models.model_vgg import build_model as build_vgg
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

# Liste des modèles à tester
model_builders = {
    "SVM": build_svm,
    "DenseNet": build_dense,
    "CNN": build_cnn,
    "VGG-Pretrained": lambda train_images, train_labels, test_images, test_labels, class_names:
        build_vgg(train_images, train_labels, test_images, test_labels, class_names, pretrained=True),
    "VGG-Untrained": lambda train_images, train_labels, test_images, test_labels, class_names:
        build_vgg(train_images, train_labels, test_images, test_labels, class_names, pretrained=False)
}

results = []

for name, build_fn in model_builders.items():
    print(f"\n================ {name} =================")

    grayscale = name != "VGG-Pretrained"
    train_images, train_labels, test_images, test_labels, class_names = load_cifar10_subset(grayscale=grayscale)

    model = build_fn(train_images, train_labels, test_images, test_labels, class_names)

    tracker = EmissionsTracker(output_dir="results", output_file=f"emissions_{name}.csv")
    tracker.start()

    start_time = time.time()
    
    model.fit()
    model.predict()
    predictions = model.pred_labels

    duration = time.time() - start_time
    emissions = tracker.stop()

    acc = accuracy_score(test_labels.ravel(), predictions)
    print(f"Accuracy: {acc:.4f}")
    print(f"Temps: {duration:.2f} sec")
    print(f"Émissions carbone: {emissions:.6f} kgCO₂eq")

    results.append({
        "model": name,
        "accuracy": acc,
        "time": duration,
        "emissions": emissions
    })

pd.DataFrame(results).to_csv("results/global_results.csv", index=False)
print("\nTous les modèles ont été testés et les résultats sont enregistrés.")
