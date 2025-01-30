import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# 1. Definicja ścieżki do pliku z danymi
path = "../Source/high_popularity_spotify_data.csv"  # dostosuj do własnej lokalizacji pliku CSV

def main():
    # 2. Wczytanie danych do DataFrame
    df = pd.read_csv(path)

    # 3. Podstawowa eksploracja (opcjonalnie)
    print("\n--- Podgląd danych ---")
    print(df.head())
    print("\n--- Informacje o danych ---")
    print(df.info())
    print("\n--- Statystyki opisowe ---")
    print(df.describe())

    # 4. Czyszczenie danych - usunięcie wierszy z brakami
    df.dropna(inplace=True)

    # 5. Tworzenie kategorii popularności na podstawie kwantyli
    #    (np. 3 kategorie: niska, średnia, wysoka)
    df['kategoria_pop'] = pd.qcut(df['track_popularity'], q=3, labels=['niska', 'średnia', 'wysoka'])

    # 6. Tworzenie kategorii innej zmiennej, np. danceability
    #    (również 3 kategorie na podstawie kwantyli)
    df['danceability_cat'] = pd.qcut(df['danceability'], q=3, labels=['niska', 'średnia', 'wysoka'])

    # -----------------------
    #   Założenia testu chi²
    # -----------------------
    # 1. Dane zebrane w sposób losowy, obserwacje są niezależne.
    # 2. Obie zmienne (kategoria_pop, danceability_cat) są kategoryczne.
    # 3. Oczekiwane liczebności w komórkach tabeli kontyngencji nie powinny być zbyt małe
    #    (najczęściej przyjmuje się wartość minimalną 5).

    # -----------------------
    #   Hipotezy statystyczne
    # -----------------------
    # H0: Zmienna kategoria_pop i danceability_cat są niezależne (brak związku).
    # H1: Istnieje zależność między kategoria_pop a danceability_cat.

    # 7. Tworzenie tabeli kontyngencji
    contingency_table = pd.crosstab(df['kategoria_pop'], df['danceability_cat'])
    print("\n--- Tabela kontyngencji (kategoria_pop vs danceability_cat) ---")
    print(contingency_table)

    # 8. Test chi-kwadrat
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    alpha = 0.05  # poziom istotności

    print("\n--- Test chi-kwadrat ---")
    print(f"Statystyka chi2 = {chi2:.4f}")
    print(f"Wartość p       = {p:.6f}")
    print(f"Liczba stopni swobody (dof) = {dof}")
    print("\n--- Tabela liczebności oczekiwanych ---")
    print(expected)

    # 9. Wnioskowanie (odrzucenie/nieodrzucenie H0)
    if p < alpha:
        print(f"\nOdrzucamy H0 na poziomie istotności α = {alpha}.")
        print("Wniosek: Istnieje statystycznie istotna zależność między kategoria_pop i danceability_cat.")
    else:
        print(f"\nBrak podstaw do odrzucenia H0 na poziomie istotności α = {alpha}.")
        print("Wniosek: Brak statystycznie istotnej zależności między kategoria_pop i danceability_cat.")

    # 10. Wizualizacja tabeli kontyngencji (mapa cieplna)
    plt.figure(figsize=(8, 6))
    sns.heatmap(contingency_table, annot=True, cmap='Blues', fmt='d')
    plt.title('Mapa cieplna: kategoria_pop vs danceability_cat')
    plt.xlabel('danceability_cat')
    plt.ylabel('kategoria_pop')
    plt.tight_layout()
    plt.savefig('chi2_heatmap_kategoria_pop_danceability_cat.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
