# System Rekomendacji Filmów - MovieLens

Projekt systemu rekomendacji filmów wykorzystujący algorytm Matrix Factorization na zbiorze danych MovieLens 100K.
System analizuje oceny użytkowników i generuje spersonalizowane rekomendacje filmów. Model został zaimplementowany w PyTorch z wykorzystaniem technik faktoryzacji macierzy oraz regularyzacji L2.

## Wymagania

```bash
pip install pandas numpy torch scikit-learn matplotlib seaborn mlflow
```

## Uruchomienie

```bash
python movielens_recommendation.py
```

## Struktura projektu

```
├── movielens_recommendation.py    # Główny skrypt
├── ml-100k/                      # Dane MovieLens
├── results/                      # Wyniki i wizualizacje
├── mlruns/                       # Eksperymenty MLflow
└── README.md
```

## Wyniki

Metryki modelu na zbiorze testowym:
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
