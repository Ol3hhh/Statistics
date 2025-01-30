import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ttest_ind, chi2_contingency, norm
import statsmodels.api as sm

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

# Wizualizacja rozkładu cech i popularności
plt.figure(figsize=(10, 6))
sns.histplot(df['track_popularity'], kde=True, color='blue')
plt.title('Rozkład popularności utworów')
plt.xlabel('Popularność')
plt.ylabel('Częstotliwość')
plt.savefig('rozkład_popularności.png', dpi=300, bbox_inches='tight')
plt.show()

# Wybrane cechy do analizy
features = ['danceability', 'energy', 'tempo', 'loudness', 'acousticness']

# Wizualizacja relacji między cechami a popularnością
for feature in features:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=df[feature], y=df['track_popularity'], alpha=0.6)
    plt.title(f'Relacja między {feature} a popularnością')
    plt.xlabel(feature)
    plt.ylabel('Popularność')
    plt.savefig(f'relacja_{feature}_popularność.png', dpi=300, bbox_inches='tight')
    plt.show()

# Korelacja między popularnością a cechami
print("\nKorelacje między cechami a popularnością:")
for feature in features:
    corr, p_value = pearsonr(df[feature], df['track_popularity'])
    print(f'{feature}: Korelacja = {corr:.2f}, wartość p = {p_value:.3f}')

# Mapa cieplna korelacji
correlation_matrix = df[['track_popularity'] + features].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Macierz korelacji')
plt.savefig('macierz_korelacji.png', dpi=300, bbox_inches='tight')
plt.show()

# Regresja liniowa dla jednej cechy
feature = 'danceability'
X = df[feature]
Y = df['track_popularity']

# Dodanie stałej do modelu
X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
print("\nPodsumowanie modelu regresji liniowej:")
print(model.summary())

# Wizualizacja regresji liniowej
plt.figure(figsize=(8, 5))
sns.regplot(x=df[feature], y=df['track_popularity'], line_kws={'color': 'red'}, ci=None)
plt.title(f'Regresja liniowa: {feature} vs popularność')
plt.xlabel(feature)
plt.ylabel('Popularność')
plt.savefig(f'regresja_{feature}_popularność.png', dpi=300, bbox_inches='tight')
plt.show()

# Test hipotezy dla dwóch grup (np. wysoka i niska popularność)
median_track_popularity = df['track_popularity'].median()
low_track_popularity = df[df['track_popularity'] <= median_track_popularity]
high_track_popularity = df[df['track_popularity'] > median_track_popularity]

print("\nTest t dla 'danceability':")
t_stat, p_val = ttest_ind(low_track_popularity['danceability'], high_track_popularity['danceability'])
print(f'Statystyka t = {t_stat:.2f}, wartość p = {p_val:.3f}')

# Test Chi-kwadrat dla niezależności
print("\nTest Chi-kwadrat dla kategorii:")
df['kategoria_pop'] = pd.qcut(df['track_popularity'], q=3, labels=['niska', 'średnia', 'wysoka'])
contingency_table = pd.crosstab(df['kategoria_pop'], df['playlist_genre'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f'Chi2 = {chi2:.2f}, wartość p = {p:.3f}')

# Oszacowanie przedziału ufności dla średniej popularności
mean_track_popularity = df['track_popularity'].mean()
std_track_popularity = df['track_popularity'].std()
n = len(df['track_popularity'])
z_score = norm.ppf(0.975)  # 95% confidence interval
margin_of_error = z_score * (std_track_popularity / (n ** 0.5))
confidence_interval = (mean_track_popularity - margin_of_error, mean_track_popularity + margin_of_error)
print("\n95% Przedział ufności dla średniej popularności:")
print(f'{confidence_interval[0]:.2f} - {confidence_interval[1]:.2f}')







# Konwersja zmiennych ciągłych na kategorie (kwantyle)
df['danceability_cat'] = pd.qcut(df['danceability'], q=3, labels=['niska', 'średnia', 'wysoka'])
df['energy_cat'] = pd.qcut(df['energy'], q=3, labels=['niska', 'średnia', 'wysoka'])
df['tempo_cat'] = pd.qcut(df['tempo'], q=3, labels=['niska', 'średnia', 'wysoka'])
df['loudness_cat'] = pd.qcut(df['loudness'], q=3, labels=['niska', 'średnia', 'wysoka'])

# Lista zmiennych do analizy
categorical_features = ['danceability_cat', 'energy_cat', 'tempo_cat', 'loudness_cat']

# Test Chi-kwadrat dla każdej cechy
chi2_results = {}
for feature in categorical_features:
    contingency_table = pd.crosstab(df['kategoria_pop'], df[feature])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    chi2_results[feature] = {'Chi2': chi2, 'p-value': p}
    print(f"\nTest Chi-kwadrat dla {feature}:")
    print(f"Chi2 = {chi2:.2f}, p-value = {p:.3f}")

# Wizualizacja map cieplnych dla tabel kontyngencji
for feature in categorical_features:
    plt.figure(figsize=(8, 6))
    contingency_table = pd.crosstab(df['kategoria_pop'], df[feature])
    sns.heatmap(contingency_table, annot=True, cmap='coolwarm', fmt='d')
    plt.title(f'Mapa cieplna: {feature} vs kategoria_pop')
    plt.xlabel(feature)
    plt.ylabel('Kategoria popularności')
    plt.savefig(f'chi2_{feature}_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

import numpy as np


# Funkcja do obliczenia Cramér's V
def cramers_v(chi2, n, k, r):
    """Oblicza miarę Cramér's V dla tabeli kontyngencji."""
    phi2 = chi2 / n
    min_dim = min(k - 1, r - 1)
    return np.sqrt(phi2 / min_dim)


# Test Chi-kwadrat dla każdej cechy i obliczenie Cramér's V
chi2_results = {}
for feature in categorical_features:
    contingency_table = pd.crosstab(df['kategoria_pop'], df[feature])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()  # liczba obserwacji
    k = contingency_table.shape[1]  # liczba kolumn
    r = contingency_table.shape[0]  # liczba wierszy
    cramers_v_value = cramers_v(chi2, n, k, r)
    chi2_results[feature] = {'Chi2': chi2, 'p-value': p, 'Cramér\'s V': cramers_v_value}

    print(f"\nTest Chi-kwadrat dla {feature}:")
    print(f"Chi2 = {chi2:.2f}, p-value = {p:.3f}, Cramér's V = {cramers_v_value:.3f}")

# Wizualizacja map cieplnych dla tabel kontyngencji
for feature in categorical_features:
    plt.figure(figsize=(8, 6))
    contingency_table = pd.crosstab(df['kategoria_pop'], df[feature])
    sns.heatmap(contingency_table, annot=True, cmap='coolwarm', fmt='d')
    plt.title(f'Mapa cieplna: {feature} vs kategoria_pop')
    plt.xlabel(feature)
    plt.ylabel('Kategoria popularności')
    plt.savefig(f'chi2_{feature}_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()


# Wnioski
print("\nWnioski:")
print("1. Korelacje wskazują, które cechy są istotne dla popularności.")
print("2. Test t pokazuje, czy różnice między grupami są statystycznie istotne.")
print("3. Test Chi-kwadrat bada zależność między kategoriami popularności a cechami kategorycznymi.")
print("4. Przedział ufności szacuje średnią popularność z określoną precyzją.")



# Lista cech numerycznych do analizy
numerical_features = df.select_dtypes(include=['float64', 'int64']).columns.drop(['track_popularity'])

# Test t-Studenta dla każdej cechy numerycznej
t_test_results = {}

print("\nTest t-Studenta dla wszystkich cech numerycznych:")
for feature in numerical_features:
    t_stat, p_val = ttest_ind(low_track_popularity[feature], high_track_popularity[feature], equal_var=False)
    t_test_results[feature] = {'t-statistic': t_stat, 'p-value': p_val}
    print(f"{feature}: t-statistic = {t_stat:.2f}, p-value = {p_val:.3f}")

# Analiza wyników
print("\nWnioski z testu t-Studenta:")
for feature, results in t_test_results.items():
    if results['p-value'] < 0.05:
        print(f"{feature}: Różnice między grupami są statystycznie istotne (p < 0.05).")
    else:
        print(f"{feature}: Brak statystycznie istotnych różnic między grupami (p >= 0.05).")
