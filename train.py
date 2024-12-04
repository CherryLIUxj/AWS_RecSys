import argparse
import os
# import pickle
import boto3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description='Train Matrix Factorization model')
    # Data and model checkpoints directories
    parser.add_argument('--bucket', type=str, default='amzrecsys')
    parser.add_argument('--train_data', type=str, default='train_set.csv')
    parser.add_argument('--test_data', type=str, default='test_set.csv')
    parser.add_argument('--ratings_data', type=str, default='ratings.csv')
    parser.add_argument('--output_dir', type=str, default='/opt/ml/model')  # save to SageMaker, and SageMaker will automatically package all the files in the /opt/ml/model directory and upload them to S3.
    parser.add_argument('--embedding_dim', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=10)
    return parser.parse_args()

class RatingDataset(Dataset):
    def __init__(self, df):
        self.user = torch.tensor(df.user_idx.values, dtype=torch.long)
        self.item = torch.tensor(df.item_idx.values, dtype=torch.long)
        self.rating = torch.tensor(df.rating.values, dtype=torch.float32)

    def __len__(self):
        return len(self.rating)

    def __getitem__(self, idx):
        return self.user[idx], self.item[idx], self.rating[idx]

class MatrixFactorization_Biased(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=20, global_bias=0.0):
        super(MatrixFactorization_Biased, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.Tensor([global_bias]))

        # Initialize embeddings and biases
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_ids, item_ids):
        user_vecs = self.user_embedding(user_ids)
        item_vecs = self.item_embedding(item_ids)
        user_b = self.user_bias(user_ids).squeeze()
        item_b = self.item_bias(item_ids).squeeze()
        dot_product = (user_vecs * item_vecs).sum(1)
        predicted_rating = self.global_bias + user_b + item_b + dot_product
        return predicted_rating

def main():
    args = parse_args()
    print("Starting training with args:", args)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create S3 client
    s3 = boto3.client('s3')

    # Download datasets from S3
    local_train_path = 'train_set.csv'
    local_test_path = 'test_set.csv'
    local_ratings_path = 'ratings.csv'
    s3.download_file(args.bucket, args.train_data, local_train_path)
    s3.download_file(args.bucket, args.test_data, local_test_path)
    s3.download_file(args.bucket, args.ratings_data, local_ratings_path)


    # Load datasets
    train_set = pd.read_csv(local_train_path)
    test_set = pd.read_csv(local_test_path)
    ratings = pd.read_csv(local_ratings_path)

    # Map user_id and item_id to indices
    users = ratings.user_id.unique()
    items = ratings.item_id.unique()
    user2idx = {user: idx for idx, user in enumerate(users)}
    item2idx = {item: idx for idx, item in enumerate(items)}

    train_set['user_idx'] = train_set.user_id.map(user2idx)
    train_set['item_idx'] = train_set.item_id.map(item2idx)
    test_set['user_idx'] = test_set.user_id.map(user2idx)
    test_set['item_idx'] = test_set.item_id.map(item2idx)
    num_users = len(users)
    num_items = len(items)

    # Save user2idx and item2idx mappings for inference
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'user2idx.pkl'), 'wb') as f:
        pickle.dump(user2idx, f)
    with open(os.path.join(args.output_dir, 'item2idx.pkl'), 'wb') as f:
        pickle.dump(item2idx, f)

    # Create datasets and dataloaders
    train_dataset = RatingDataset(train_set)
    test_dataset = RatingDataset(test_set)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Initialize model
    global_bias = train_set['rating'].mean()
    model = MatrixFactorization_Biased(num_users, num_items, embedding_dim=args.embedding_dim, global_bias=global_bias).to(device)

    # Define loss and optimizer
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Training loop
    best_test_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for user_ids, item_ids, ratings in train_loader:
            user_ids, item_ids, ratings = user_ids.to(device), item_ids.to(device), ratings.to(device)
            optimizer.zero_grad()
            predictions = model(user_ids, item_ids)
            loss = loss_func(predictions, ratings)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for user_ids, item_ids, ratings in test_loader:
                user_ids, item_ids, ratings = user_ids.to(device), item_ids.to(device), ratings.to(device)
                predictions = model(user_ids, item_ids)
                loss = loss_func(predictions, ratings)
                test_loss += loss.item()
        avg_test_loss = test_loss / len(test_loader)

        print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')

        # Early stopping
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            epochs_no_improve = 0
            # Save the best model
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'mf_model.pth'))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print('Early stopping!')
                break

        scheduler.step()

    print('Training completed.')

if __name__ == '__main__':
    main()









