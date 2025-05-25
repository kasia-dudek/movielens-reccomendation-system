# System Rekomendacji Filmów - MovieLens

Projekt systemu rekomendacji filmów wykorzystujący algorytm Matrix Factorization na zbiorze danych MovieLens 100K.

## Opis projektu

System analizuje oceny użytkowników i generuje spersonalizowane rekomendacje filmów. Model został zaimplementowany w PyTorch z wykorzystaniem technik faktoryzacji macierzy oraz regularyzacji L2.

## Funkcjonalności

- **Analiza danych**: Wczytywanie i preprocessing zbioru MovieLens 100K
- **Model ML**: Matrix Factorization z bias terms i dropout
- **Strojenie parametrów**: Automatyczne wyszukiwanie optymalnych hiperparametrów
- **Ewaluacja**: Metryki RMSE, MAE, Precision@K, Recall@K, F1@K
- **Wizualizacje**: Wykresy rozkładu ocen, krzywej uczenia, macierzy pomyłek
- **Rekomendacje**: Generowanie top-K filmów dla użytkowników
- **Tracking**: Integracja z MLflow do śledzenia eksperymentów

## Wymagania

```bash
pip install pandas numpy torch scikit-learn matplotlib seaborn mlflow
```

## Uruchomienie

1. Pobierz dane MovieLens 100K i rozpakuj do folderu `ml-100k/`
2. Uruchom główny skrypt:

```bash
python movielens_recommendation.py
```

## Struktura projektu

```
├── movielens_recommendation.py    # Główny skrypt
├── ml-100k/                      # Dane MovieLens (ignorowane w git)
├── results/                      # Wyniki i wizualizacje (ignorowane w git)
├── mlruns/                       # Eksperymenty MLflow (ignorowane w git)
└── README.md
```

## Wyniki

Model osiąga następujące metryki na zbiorze testowym:
- **RMSE**: 0.931
- **MAE**: 0.733
- **Precision@5**: 74.1%
- **Recall@5**: 46.3%

Najlepsze parametry: embedding_dim=32, l2_lambda=0.0001

## Pliki wyjściowe

Po uruchomieniu skryptu generowane są:
- `results/best_model.pth` - wytrenowany model
- `results/evaluation_results.csv` - metryki ewaluacji
- `results/recommendations.csv` - przykładowe rekomendacje
- Różne wykresy w formacie PNG

## Uwagi

- Dane MovieLens nie są dołączone do repozytorium ze względu na rozmiar
- Eksperymenty MLflow są zapisywane lokalnie w folderze `mlruns/`
- Model można łatwo dostosować zmieniając parametry w funkcji `hyperparameter_tuning()`
