import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Wczytanie danych
path = "../Source/high_popularity_spotify_data.csv"
df = pd.read_csv(path)

# Eksploracja danych
print("Podgląd danych:")
print(df.head())
print("\nInformacje o danych:")
print(df.info())
print("\nStatystyki opisowe:")
print(df.describe())

# Czyszczenie danych - usunięcie brakujących wartości
df.dropna(inplace=True)

# Podział danych na grupy: niska i wysoka popularność
median_track_popularity = df['track_popularity'].median()
low_track_popularity = df[df['track_popularity'] <= median_track_popularity]
high_track_popularity = df[df['track_popularity'] > median_track_popularity]

# Formułowanie hipotez
print("\nHipotezy:")
print("H0: Średnia wartość danceability dla utworów o niskiej popularności jest równa średniej wartości danceability dla utworów o wysokiej popularności.")
print("H1: Średnia wartość danceability dla utworów o niskiej popularności jest różna od średniej wartości danceability dla utworów o wysokiej popularności.")

# Test t-Studenta dla cechy 'danceability'
print("\nTest t-Studenta dla 'danceability':")
t_stat, p_val = ttest_ind(low_track_popularity['danceability'], high_track_popularity['danceability'], equal_var=False)
print(f'Statystyka t = {t_stat:.2f}, wartość p = {p_val:.3f}')

# Interpretacja wyników
alpha = 0.05  # Przyjęty poziom istotności
if p_val < alpha:
    print("\nWniosek: Odrzucamy hipotezę H0 na rzecz H1. Istnieje statystycznie istotna różnica w średniej wartości danceability między utworami o niskiej i wysokiej popularności.")
else:
    print("\nWniosek: Nie ma wystarczających dowodów, aby odrzucić hipotezę H0. Brak statystycznie istotnej różnicy w średniej wartości danceability między utworami o niskiej i wysokiej popularności.")

# Wizualizacja rozkładu danceability w obu grupach
plt.figure(figsize=(10, 6))
sns.histplot(low_track_popularity['danceability'], color='blue', label='Niska popularność', kde=True)
sns.histplot(high_track_popularity['danceability'], color='red', label='Wysoka popularność', kde=True)
plt.title('Rozkład danceability dla utworów o niskiej i wysokiej popularności')
plt.xlabel('Danceability')
plt.ylabel('Częstotliwość')
plt.legend()
plt.savefig('../Graphs/H/rozkład_danceability.png', dpi=300, bbox_inches='tight')
plt.show()


# Funkcja do testowania hipotez
def test_hipotez(feature, low_group, high_group, alpha=0.05):
    print(f"\nTest t-Studenta dla '{feature}':")
    t_stat, p_val = ttest_ind(low_group[feature], high_group[feature], equal_var=False)
    print(f'Statystyka t = {t_stat:.2f}, wartość p = {p_val:.3f}')

    if p_val < alpha:
        print(
            f"Wniosek: Odrzucamy H0 na rzecz H1. Istnieje statystycznie istotna różnica w średniej wartości {feature} między grupami.")
    else:
        print(
            f"Wniosek: Nie ma wystarczających dowodów, aby odrzucić H0. Brak statystycznie istotnej różnicy w średniej wartości {feature} między grupami.")


# Testowanie hipotez dla loudness, tempo i energy
features_to_test = ['loudness', 'tempo', 'energy']
for feature in features_to_test:
    print(f"\nHipotezy dla '{feature}':")
    print(
        f"H0: Średnia wartość {feature} dla utworów o niskiej popularności jest równa średniej wartości {feature} dla utworów o wysokiej popularności.")
    print(
        f"H1: Średnia wartość {feature} dla utworów o niskiej popularności jest różna od średniej wartości {feature} dla utworów o wysokiej popularności.")
    test_hipotez(feature, low_track_popularity, high_track_popularity)



# Wizualizacja rozkładów dla loudness, tempo i energy
for feature in features_to_test:
    plt.figure(figsize=(10, 6))
    sns.histplot(low_track_popularity[feature], color='blue', label='Niska popularność', kde=True)
    sns.histplot(high_track_popularity[feature], color='red', label='Wysoka popularność', kde=True)
    plt.title(f'Rozkład {feature} dla utworów o niskiej i wysokiej popularności')
    plt.xlabel(feature)
    plt.ylabel('Częstotliwość')
    plt.legend()
    plt.savefig(f'../Graphs/H/rozkład_{feature}.png', dpi=300, bbox_inches='tight')
    plt.show()