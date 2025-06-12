import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import mlflow
import itertools

# Krok 1: Wczytanie danych MovieLens 100K
def load_data(data_path):
    columns = ['userId', 'movieId', 'rating', 'timestamp']
    data = pd.read_csv(data_path, sep='\t', names=columns)
    print("Przykładowe dane:")
    print(data.head())
    print("\nStatystyki danych:")
    print(data.describe())
    print(f"Liczba unikalnych użytkowników: {data['userId'].nunique()}")
    print(f"Liczba unikalnych filmów: {data['movieId'].nunique()}")
    return data

# Krok 2: Preprocessing danych
def preprocess_data(data):
    user_id_map = {id: idx for idx, id in enumerate(data['userId'].unique())}
    movie_id_map = {id: idx for idx, id in enumerate(data['movieId'].unique())}
    
    data['user_idx'] = data['userId'].map(user_id_map)
    data['movie_idx'] = data['movieId'].map(movie_id_map)
    
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    train_users = torch.tensor(train_data['user_idx'].values, dtype=torch.long)
    train_items = torch.tensor(train_data['movie_idx'].values, dtype=torch.long)
    train_ratings = torch.tensor(train_data['rating'].values, dtype=torch.float32)
    
    test_users = torch.tensor(test_data['user_idx'].values, dtype=torch.long)
    test_items = torch.tensor(test_data['movie_idx'].values, dtype=torch.long)
    test_ratings = torch.tensor(test_data['rating'].values, dtype=torch.float32)
    
    print(f"Rozmiar zbioru treningowego: {len(train_data)}")
    print(f"Rozmiar zbioru testowego: {len(test_data)}")
    
    return (train_users, train_items, train_ratings), (test_users, test_items, test_ratings), user_id_map, movie_id_map

# Krok 3: Wizualizacja rozkładu ocen
def plot_rating_distribution(data, output_dir):
    plt.figure(figsize=(10, 6), dpi=100)
    plt.hist(data['rating'], bins=5, edgecolor='black', alpha=0.7)
    plt.title('Rozkład ocen w MovieLens 100K')
    plt.xlabel('Ocena')
    plt.ylabel('Liczba ocen')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'rating_distribution.png'), bbox_inches='tight')
    plt.close()

# Krok 4: Definicja ulepszonego modelu z biasami i dropout
class MFModelWithBias(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=50, dropout_rate=0.2):
        super(MFModelWithBias, self).__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(dropout_rate)
        
        self.user_embeddings.weight.data.uniform_(-0.01, 0.01)
        self.item_embeddings.weight.data.uniform_(-0.01, 0.01)
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)
    
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embeddings(user_ids)
        item_emb = self.item_embeddings(item_ids)
        user_emb = self.dropout(user_emb)
        item_emb = self.dropout(item_emb)
        user_b = self.user_bias(user_ids).squeeze()
        item_b = self.item_bias(item_ids).squeeze()
        return (user_emb * item_emb).sum(1) + user_b + item_b + self.global_bias

# Krok 5: Trening modelu z regularyzacją L2
def train_model(model, train_data, num_epochs=10, batch_size=256, l2_lambda=1e-5):
    train_users, train_items, train_ratings = train_data
    train_dataset = TensorDataset(train_users, train_items, train_ratings)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=l2_lambda)
    
    losses = []
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for user_ids, item_ids, ratings in train_loader:
            optimizer.zero_grad()
            predictions = model(user_ids, item_ids)
            loss = criterion(predictions, ratings)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return losses

# Krok 6: Wizualizacja straty treningu
def plot_training_loss(losses, output_dir):
    plt.figure(figsize=(10, 6), dpi=100)
    plt.plot(range(1, len(losses) + 1), losses, marker='o', color='b')
    plt.title('Strata treningu w kolejnych epokach')
    plt.xlabel('Epoka')
    plt.ylabel('Strata (MSE)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_loss.png'), bbox_inches='tight')
    plt.close()

# Krok 7: Wizualizacja przewidywanych vs rzeczywistych ocen
def plot_predicted_vs_actual(test_data, predictions, output_dir):
    test_ratings = test_data[2].numpy()
    plt.figure(figsize=(10, 6), dpi=100)
    plt.scatter(test_ratings, predictions, alpha=0.5, color='r')
    plt.plot([1, 5], [1, 5], 'k--')  # Linia idealnego dopasowania
    plt.title('Przewidywane vs rzeczywiste oceny')
    plt.xlabel('Rzeczywiste oceny')
    plt.ylabel('Przewidywane oceny')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'predicted_vs_actual.png'), bbox_inches='tight')
    plt.close()

# Krok 8: Macierz pomyłek dla zaokrąglonych ocen
def plot_confusion_matrix(test_data, predictions, output_dir):
    test_ratings = test_data[2].numpy()
    predicted_ratings = np.clip(np.round(predictions), 1, 5)  # Zaokrąglenie do 1-5
    cm = confusion_matrix(test_ratings, predicted_ratings)
    plt.figure(figsize=(8, 6), dpi=100)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Macierz pomyłek dla zaokrąglonych ocen')
    plt.xlabel('Przewidywana ocena')
    plt.ylabel('Rzeczywista ocena')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), bbox_inches='tight')
    plt.close()

# Krok 9: Ewaluacja modelu z dodatkowymi metrykami
def evaluate_model(model, test_data):
    test_users, test_items, test_ratings = test_data
    model.eval()
    with torch.no_grad():
        predictions = model(test_users, test_items)
        rmse = np.sqrt(mean_squared_error(test_ratings.numpy(), predictions.numpy()))
        mae = mean_absolute_error(test_ratings.numpy(), predictions.numpy())
        
        # Precision@K, Recall@K, F1@K dla różnych K
        k_values = [5, 10]
        metrics = {'precision': {}, 'recall': {}, 'f1': {}}
        for k in k_values:
            top_k_preds = []
            top_k_recalls = []
            for user in torch.unique(test_users):
                user_mask = test_users == user
                user_preds = predictions[user_mask]
                user_true = test_ratings[user_mask]
                
                if len(user_preds) >= k:
                    top_k_indices = torch.topk(user_preds, k).indices
                    top_k_true = user_true[top_k_indices]
                    relevant = (top_k_true >= 4.0).float().sum().item()
                    precision_at_k = relevant / k
                    recall_at_k = relevant / len(user_true[user_true >= 4.0]) if len(user_true[user_true >= 4.0]) > 0 else 0.0
                    f1_at_k = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k) if (precision_at_k + recall_at_k) > 0 else 0.0
                    top_k_preds.append(precision_at_k)
                    top_k_recalls.append(recall_at_k)
            metrics['precision'][k] = np.mean(top_k_preds) if top_k_preds else 0.0
            metrics['recall'][k] = np.mean(top_k_recalls) if top_k_recalls else 0.0
            metrics['f1'][k] = 2 * (metrics['precision'][k] * metrics['recall'][k]) / (metrics['precision'][k] + metrics['recall'][k]) if (metrics['precision'][k] + metrics['recall'][k]) > 0 else 0.0
    
    return rmse, mae, metrics, predictions.numpy()

# Krok 10: Wizualizacja Precision@K dla różnych K
def plot_precision_recall(metrics, output_dir):
    k_values = list(metrics['precision'].keys())
    precisions = [metrics['precision'][k] for k in k_values]
    recalls = [metrics['recall'][k] for k in k_values]
    
    plt.figure(figsize=(10, 6), dpi=100)
    plt.plot(k_values, precisions, marker='o', label='Precision@K')
    plt.plot(k_values, recalls, marker='s', label='Recall@K')
    plt.title('Precision@K i Recall@K dla różnych K')
    plt.xlabel('K')
    plt.ylabel('Wartość')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'precision_recall.png'), bbox_inches='tight')
    plt.close()

# Krok 11: Generowanie rekomendacji dla wielu użytkowników
def recommend_movies(model, user_id, user_id_map, movie_id_map, num_items, top_k=5):
    user_idx = user_id_map[user_id]
    user_tensor = torch.tensor([user_idx] * num_items, dtype=torch.long)
    item_tensor = torch.tensor(range(num_items), dtype=torch.long)
    
    model.eval()
    with torch.no_grad():
        scores = model(user_tensor, item_tensor)
        top_indices = torch.topk(scores, k=top_k).indices.numpy()
    
    reverse_movie_map = {idx: id for id, idx in movie_id_map.items()}
    recommended_movie_ids = [reverse_movie_map[idx] for idx in top_indices]
    return recommended_movie_ids, scores[top_indices].numpy()

# Krok 12: Wizualizacja rekomendacji
def plot_recommendations(user_id, recommended_movie_ids, scores, output_dir):
    plt.figure(figsize=(10, 6), dpi=100)
    plt.bar(range(len(recommended_movie_ids)), scores, tick_label=[f"Movie {id}" for id in recommended_movie_ids])
    plt.title(f'Rekomendacje dla użytkownika {user_id}')
    plt.xlabel('ID filmu')
    plt.ylabel('Przewidywana ocena')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'recommendations_user_{user_id}.png'), bbox_inches='tight')
    plt.close()

# Krok 13: Strojenie hiperparametrów
def hyperparameter_tuning(train_data, test_data, num_users, num_items, output_dir):
    embedding_dims = [32, 50, 64]
    l2_lambdas = [1e-5, 1e-4, 1e-3]
    
    best_rmse = float('inf')
    best_model = None
    best_params = None
    
    for emb_dim, l2_lambda in itertools.product(embedding_dims, l2_lambdas):
        print(f"\nTestowanie: embedding_dim={emb_dim}, l2_lambda={l2_lambda}")
        model = MFModelWithBias(num_users, num_items, embedding_dim=emb_dim)
        losses = train_model(model, train_data, num_epochs=10, l2_lambda=l2_lambda)
        rmse, mae, metrics, predictions = evaluate_model(model, test_data)
        
        mlflow.log_param(f"embedding_dim_{emb_dim}_{l2_lambda}", emb_dim)
        mlflow.log_param(f"l2_lambda_{emb_dim}_{l2_lambda}", l2_lambda)
        mlflow.log_metric(f"rmse_{emb_dim}_{l2_lambda}", rmse)
        mlflow.log_metric(f"mae_{emb_dim}_{l2_lambda}", mae)
        for k in metrics['precision']:
            mlflow.log_metric(f"precision_at_{k}_{emb_dim}_{l2_lambda}", metrics['precision'][k])
            mlflow.log_metric(f"recall_at_{k}_{emb_dim}_{l2_lambda}", metrics['recall'][k])
            mlflow.log_metric(f"f1_at_{k}_{emb_dim}_{l2_lambda}", metrics['f1'][k])
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_params = {'embedding_dim': emb_dim, 'l2_lambda': l2_lambda}
    
    mlflow.log_param("best_embedding_dim", best_params['embedding_dim'])
    mlflow.log_param("best_l2_lambda", best_params['l2_lambda'])
    mlflow.log_metric("best_rmse", best_rmse)
    
    print(f"\nNajlepsze parametry: {best_params}, RMSE: {best_rmse:.4f}")
    return best_model, best_params

# Krok 14: Główna funkcja
def main():
    # Ścieżki
    data_path = 'ml-100k/u.data'
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Wczytanie danych
    data = load_data(data_path)
    
    # Wizualizacja rozkładu ocen
    plot_rating_distribution(data, output_dir)
    
    # Preprocessing
    train_data, test_data, user_id_map, movie_id_map = preprocess_data(data)
    
    # Strojenie hiperparametrów
    num_users = len(user_id_map)
    num_items = len(movie_id_map)
    with mlflow.start_run(run_name="movielens_experiment"):
        best_model, best_params = hyperparameter_tuning(train_data, test_data, num_users, num_items, output_dir)
        
        # Ewaluacja najlepszego modelu
        rmse, mae, metrics, predictions = evaluate_model(best_model, test_data)
        print(f"\nWyniki ewaluacji najlepszego modelu:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        for k in metrics['precision']:
            print(f"Precision@{k}: {metrics['precision'][k]:.4f}")
            print(f"Recall@{k}: {metrics['recall'][k]:.4f}")
            print(f"F1@{k}: {metrics['f1'][k]:.4f}")
        
        # Zapis wyników ewaluacji
        eval_results = pd.DataFrame({
            'Metric': ['RMSE', 'MAE'] + [f'Precision@{k}' for k in metrics['precision']] + [f'Recall@{k}' for k in metrics['recall']] + [f'F1@{k}' for k in metrics['f1']],
            'Value': [rmse, mae] + [metrics['precision'][k] for k in metrics['precision']] + [metrics['recall'][k] for k in metrics['recall']] + [metrics['f1'][k] for k in metrics['f1']]
        })
        eval_results.to_csv(os.path.join(output_dir, 'evaluation_results.csv'), index=False)
        
        # Wizualizacje ewaluacyjne
        plot_predicted_vs_actual(test_data, predictions, output_dir)
        plot_confusion_matrix(test_data, predictions, output_dir)
        plot_precision_recall(metrics, output_dir)
        
        # Generowanie i zapis rekomendacji dla wielu użytkowników
        recommendation_data = []
        for user_id in [1, 10, 100]:  # Przykładowi użytkownicy
            recommended_movie_ids, scores = recommend_movies(best_model, user_id, user_id_map, movie_id_map, num_items, top_k=5)
            print(f"\nRekomendacje dla użytkownika {user_id}: {recommended_movie_ids}")
            plot_recommendations(user_id, recommended_movie_ids, scores, output_dir)
            recommendation_data.extend([{'user_id': user_id, 'movie_id': movie_id, 'score': score} 
                                       for movie_id, score in zip(recommended_movie_ids, scores)])
        
        recommendation_df = pd.DataFrame(recommendation_data)
        recommendation_df.to_csv(os.path.join(output_dir, 'recommendations.csv'), index=False)
        
        # Zapis modelu
        torch.save(best_model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
        print(f"Model zapisany w {os.path.join(output_dir, 'best_model.pth')}")
        
        # Logowanie artefaktów do MLflow
        mlflow.log_artifact(output_dir)

if __name__ == "__main__":
    main()
