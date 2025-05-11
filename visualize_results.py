import pandas as pd
import matplotlib.pyplot as plt

# Chargement des données
df = pd.read_csv("results/global_results.csv")

# Graphique 1 - Accuracy
plt.figure(figsize=(8, 5))
plt.bar(df['model'], df['accuracy'], color='cornflowerblue')
plt.title("Accuracy de chaque modèle")
plt.ylabel("Accuracy")
plt.xlabel("Modèle")
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("results/graph_accuracy.png")
plt.show()

# Graphique 2 - Temps d'exécution
plt.figure(figsize=(8, 5))
plt.bar(df['model'], df['time'], color='mediumseagreen')
plt.title("Temps d'entraînement par modèle")
plt.ylabel("Temps (secondes)")
plt.xlabel("Modèle")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("results/graph_time.png")
plt.show()

# Graphique 3 - Émissions carbone
plt.figure(figsize=(8, 5))
plt.bar(df['model'], df['emissions'], color='salmon')
plt.title("Émissions carbone par modèle")
plt.ylabel("Émissions (kg CO₂eq)")
plt.xlabel("Modèle")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("results/graph_emissions.png")
plt.show()
