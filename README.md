# System Rekomendacji Filmów - MovieLens

Prace przygotowały:
Karolina Dekajło i Katarzyna Dudek

## 1. Streszczenie
Projekt dotyczy stworzenia systemu rekomendacji filmów opartego na algorytmie Matrix Factorization, wykorzystującym zbiór danych MovieLens 100K. Celem projektu było zaimplementowanie modelu w PyTorch, który na podstawie ocen użytkowników generuje spersonalizowane rekomendacje filmów. System uwzględnia regularyzację L2 oraz różne techniki optymalizacji w celu poprawy dokładności przewidywań.

## 2. Wprowadzenie
Współczesne systemy rekomendacji są kluczowym elementem wielu aplikacji internetowych, takich jak platformy streamingowe, sklepy internetowe czy serwisy społecznościowe. Dzięki tym systemom użytkownicy otrzymują spersonalizowane propozycje, co poprawia ich doświadczenia i zwiększa zaangażowanie. Projekt ten koncentruje się na stworzeniu systemu rekomendacji filmów, który analizuje oceny użytkowników na platformie MovieLens i generuje rekomendacje na podstawie tych danych. Model oparty jest na metodzie faktoryzacji macierzy, jednej z popularniejszych technik wykorzystywanych w rekomendacjach.

## 3. Krótki przegląd literatury dotyczący tematyki projektu
Faktoryzacja macierzy jest jedną z najbardziej popularnych technik wykorzystywanych w systemach rekomendacji, ponieważ pozwala na odkrywanie ukrytych wzorców w danych. Metody takie jak Singular Value Decomposition (SVD) czy Alternating Least Squares (ALS) są szeroko stosowane w kontekście rekomendacji filmowych. Matrix Factorization (MF) polega na dekompozycji macierzy ocen użytkowników w celu uzyskania ukrytych czynników, które opisują preferencje użytkowników oraz cechy filmów. W projekcie zastosowano ulepszoną wersję MF, dodającą biasy oraz dropout w celu poprawy ogólnej wydajności modelu.

## 4. Opis danych
Dane pochodzą z zestawu MovieLens 100K, który zawiera 100 000 ocen użytkowników dla 1 682 filmów. Dane zawierają informacje o identyfikatorach użytkowników, identyfikatorach filmów, ocenach oraz znacznikach czasowych. W projekcie wykorzystano jedynie ocenę (rating) użytkowników do trenowania modelu rekomendacji, przy czym każdy użytkownik i film są mapowani na indeksy numeryczne, aby mogły być wykorzystane w modelu opartym na Tensorach.

## 5. Działanie modelu
- **Wczytanie i preprocessing danych** – Dane zostały wczytane z pliku CSV, a identyfikatory użytkowników i filmów zostały zmapowane na unikalne indeksy numeryczne.
- **Modelowanie** – Zaimplementowano model rekomendacji oparty na faktoryzacji macierzy z dodatkowymi warstwami biasów oraz mechanizmem dropout, co miało na celu poprawienie ogólnej wydajności modelu.
- **Trening i ewaluacja** – Model był trenowany przy użyciu algorytmu Adam, z regularyzacją L2 w celu uniknięcia nadmiernego dopasowania. Ewaluacja modelu przeprowadzona została za pomocą miar takich jak RMSE, MAE, Precision@K, Recall@K oraz F1@K.
- **Optymalizacja hiperparametrów** – W ramach optymalizacji testowane były różne wartości rozmiaru wektora osadzania (embedding_dim) oraz współczynnika regularyzacji L2. Najlepszy zestaw hiperparametrów uzyskano dla embedding_dim=32 i l2_lambda=0.0001.
- **Rekomendacje** – Dla kilku użytkowników wygenerowane zostały rekomendacje filmów, a wyniki zapisano do pliku CSV. Dodatkowo, wygenerowane zostały wizualizacje, takie jak wykresy przewidywanych ocen w porównaniu do rzeczywistych ocen oraz macierz pomyłek.
- **Logowanie eksperymentów** – Eksperymenty zostały zapisane przy pomocy MLflow, co umożliwia śledzenie wyników oraz parametrów modelu.

## 6. Wyniki
Metryki modelu na zbiorze testowym:
- **RMSE**: 0.931
- **MAE**: 0.733
- **Precision@5**: 74.1%
- **Recall@5**: 46.3%

Najlepsze parametry: embedding_dim=32, l2_lambda=0.0001

### Rozkład ocen w MovieLens
![rating_distribution](https://github.com/user-attachments/assets/dc4ef01c-40d1-4ce9-80f9-12edf6d055a3)

### Wykres straty treningu
![training_loss](https://github.com/user-attachments/assets/99882539-456c-4f49-9b2b-a3b612fbf45d)

### Przewidywane vs rzeczywiste oceny
![predicted_vs_actual](https://github.com/user-attachments/assets/bf8cfc68-fc12-48b1-97a1-164095c012e3)

### Macierz pomyłek
![confusion_matrix](https://github.com/user-attachments/assets/b9298e04-d028-4292-a8ea-4a173097c085)

### Rekomendacje dla trzech użytkowników
![image](https://github.com/user-attachments/assets/2e287fc2-747f-4c85-94c4-ce131b3edeea)

## 7. Struktura i uruchomienie kodu
### Wymagania
```bash
pip install pandas numpy torch scikit-learn matplotlib seaborn mlflow
```

### Uruchomienie
```bash
python movielens_recommendation.py
```

### Struktura projektu
```
├── movielens_recommendation.py   # Główny skrypt
├── ml-100k/                      # Dane MovieLens
├── results/                      # Wyniki i wizualizacje
├── mlruns/                       # Eksperymenty MLflow
└── README.md                     # Plik z tym sprawozdaniem
```

### Pliki wyjściowe
Po uruchomieniu skryptu generowane są:
- `results/best_model.pth` - wytrenowany model
- `results/evaluation_results.csv` - metryki ewaluacji
- `results/recommendations.csv` - przykładowe rekomendacje

## 8. Proces wdrożenia na Azure Machine Learning
Aby wdrożyć ten model rekomendacji na platformie Azure Machine Learning, należy przejść przez następujące etapy:
1. **Utworzenie zasobu Azure Machine Learning Workspace**
   - W pierwszym kroku utworzyć zasób Azure Machine Learning Workspace w portalu Azure, który stanowi centralne miejsce do zarządzania wszystkimi zasobami związanymi z modelem.
2. **Rejestracja modelu w Model Registry**
   - Po zakończeniu treningu lokalnie, model powinien być zarejestrowany w **Model Registry**, co umożliwia zarządzanie wersjami i udostępnianie modelu do późniejszych wdrożeń.
3. **Wdrożenie modelu jako usługę webową (REST API)**
   - Model mógłby zostać wdrożony jako usługa webowa poprzez utworzenie **REST API**, co umożliwiło łatwą interakcję z modelem i jego integrację z innymi aplikacjami. Dzięki temu użytkownicy mogli w prosty sposób wysyłać zapytania do modelu i otrzymywać rekomendacje filmów.
4. **Testowanie API i monitorowanie jego działania**
   - Po wdrożeniu, API powinno być przetestowane, a jego działanie monitorowane w czasie rzeczywistym. Azure ML oferuje narzędzia do monitorowania wydajności i dokładności modelu, co pozwala na bieżąco analizować jego efektywność.
5. **Aktualizacja i zarządzanie wersjami modelu**
   - W przypadku poprawy wydajności modelu, wprowadzania nowych technik, bądź zmiany hiperparametrów, możliwe jest zarządzanie wersjami modelu w Azure Machine Learning. Każda nowa wersja modelu może być zarejestrowana i wdrożona, zachowując kontrolę nad jego ewolucją.
  
Wyniki ewaluacji wskazują, że model uzyskał dobry poziom dokładności w przewidywaniu ocen, z RMSE na poziomie 0.931 oraz MAE wynoszącym 0.733. Precision@5 wyniosło 74.1%, a Recall@5 to 46.3%, co wskazuje na efektywność modelu w dostarczaniu trafnych rekomendacji. Jednakże, rozkład ocen w danych treningowych nie był równomierny, co mogło wpłynąć na modelowanie i wyniki, szczególnie w przypadku ocen 3 i 4. Pomimo tego, model nadal dobrze radzi sobie z generowaniem sensownych rekomendacji. Strojenie hiperparametrów oraz zastosowanie regularyzacji L2 i dropout przyczyniły się do poprawy wydajności modelu i uniknięcia nadmiernego dopasowania.
