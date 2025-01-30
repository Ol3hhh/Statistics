import os
import pandas as pd
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Wczytanie danych
path = "../Source/high_popularity_spotify_data.csv"  # Upewnij się, że ścieżka jest poprawna
df = pd.read_csv(path)

# Ścieżka do folderu zapisu wykresów
output_folder = "../Graphs"

# Upewnij się, że folder istnieje
os.makedirs(output_folder, exist_ok=True)

# Wybrane cechy do analizy rozkładu normalnego
selected_features = ['danceability', 'energy', 'tempo', 'loudness']

# Wizualizacja rozkładu normalnego dla każdej cechy
for feature in selected_features:
    plt.figure(figsize=(8, 6))

    # Rysowanie histogramu cechy z KDE
    sns.histplot(df[feature], kde=True, stat="density", color="blue", alpha=0.6)

    # Obliczanie dopasowanej krzywej normalnej
    mean, std = df[feature].mean(), df[feature].std()
    x = np.linspace(df[feature].min(), df[feature].max(), 100)
    plt.plot(x, norm.pdf(x, mean, std), color="red", label=f'Normal Fit\n$\mu={mean:.2f},\ \sigma={std:.2f}$')

    # Ustawienia wykresu
    plt.title(f'Rozkład normalny dla cechy: {feature}')
    plt.xlabel(feature)
    plt.ylabel('Gęstość')
    plt.legend()

    # Zapis wykresu do folderu ../Graphs
    output_path = os.path.join(output_folder, f'rozklad_normalny_{feature}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
