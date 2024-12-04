import os
import pickle
import boto3
import torch
import argparse
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import heapq

# Define the model class
class MatrixFactorization_Biased(torch.nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=20, global_bias=0.0):
        super(MatrixFactorization_Biased, self).__init__()
        self.user_embedding = torch.nn.Embedding(num_users, embedding_dim)
        self.item_embedding = torch.nn.Embedding(num_items, embedding_dim)
        self.user_bias = torch.nn.Embedding(num_users, 1)
        self.item_bias = torch.nn.Embedding(num_items, 1)
        self.global_bias = torch.nn.Parameter(torch.Tensor([global_bias]))

    def forward(self, user_ids, item_ids):
        user_vecs = self.user_embedding(user_ids)
        item_vecs = self.item_embedding(item_ids)
        user_b = self.user_bias(user_ids).squeeze()
        item_b = self.item_bias(item_ids).squeeze()
        dot_product = (user_vecs * item_vecs).sum(1)
        predicted_rating = self.global_bias + user_b + item_b + dot_product
        return predicted_rating
    
def parse_args():
    parser = argparse.ArgumentParser(description='Serve the recommendation model')
    parser.add_argument('--model_dir', type=str, default='/opt/ml/model',
                        help='Directory where the model artifacts are stored')
    parser.add_argument('--bucket', type=str, default='amzrecsys',
                        help='S3 bucket where the item metadata and historical data are stored')
    parser.add_argument('--item_metadata', type=str, default='product.csv',
                        help='S3 key for the item metadata CSV file')
    parser.add_argument('--historical_data', type=str, default='train_set.csv',
                        help='S3 key for the historical interaction data CSV file')
    return parser.parse_args()

# Initialize the Flask app
app = Flask(__name__)

# Load model and artifacts
def load_model():
    args = parse_args()
    model_dir = args.model_dir  # SageMaker Directory

    # Load user2idx and item2idx mappings
    with open(os.path.join(model_dir, 'user2idx.pkl'), 'rb') as f:
        user2idx = pickle.load(f)
    with open(os.path.join(model_dir, 'item2idx.pkl'), 'rb') as f:
        item2idx = pickle.load(f)

    num_users = len(user2idx)
    num_items = len(item2idx)

    idx_to_item = {idx: item_id for item_id, idx in item2idx.items()}

    # Load the trained model dict
    model_path = os.path.join(model_dir, 'mf_model.pth')
    model_dict = torch.load(model_path, map_location=torch.device('cpu'))

    # Create S3 client
    s3 = boto3.client('s3')

    # Download datasets from S3
    local_metadata_path = 'metadata.csv'
    local_hisdata_path = 'hisdata.csv'
    s3.download_file(args.bucket, args.item_metadata, local_metadata_path)
    s3.download_file(args.bucket, args.historical_data, local_hisdata_path)

    item_metadata = pd.read_csv(local_metadata_path)
    historical_data = pd.read_csv(local_hisdata_path)

    if 'title' not in item_metadata.columns:
        raise ValueError("Item metadata must contain a 'title' column.")
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(item_metadata['title'])

    return {
        'model_dict': model_dict,
        'user2idx': user2idx,
        'item2idx': item2idx,
        'idx_to_item': idx_to_item,
        'item_metadata': item_metadata,
        'historical_data': historical_data,
        'tfidf_vectorizer': tfidf_vectorizer
    }

# Load the model and artifacts at startup
context = load_model()

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'message': 'Ping successful!'}), 200

@app.route('/invocations', methods=['POST'])
def invocations():
    input_data = request.get_json()
    user_id = input_data.get('user_id')
    top_k = input_data.get('top_k', 100)

    if user_id is None:
        return jsonify({'error': 'user_id not provided'}), 400

    # Generate recommendations
    predictions = generate_recommendations(user_id, top_k)
    if 'error' in predictions:
        return jsonify(predictions), 400
    else:
        return jsonify(predictions), 200
    

def generate_recommendations(user_id, top_k=100):
    model_dict = context['model_dict']
    user2idx = context['user2idx']
    item2idx = context['item2idx']
    idx_to_item = context['idx_to_item']
    item_metadata = context['item_metadata']
    historical_data = context['historical_data']
    tfidf_vectorizer = context['tfidf_vectorizer']
    
    # Check if user exists
    if user_id not in user2idx:
        return {'error': f'User ID {user_id} not found'}
    
    # Access embeddings and biases
    user_embeddings = model_dict['user_embedding.weight'].numpy()  ## model['user_embedding.weight'] tensor already on cpu, no need to detach
    item_embeddings = model_dict['item_embedding.weight'].numpy()
    user_biases = model_dict['user_bias.weight'].numpy().squeeze()
    item_biases = model_dict['item_bias.weight'].numpy().squeeze()
    global_bias = model_dict['global_bias'].item()

    # Prepare user vector
    user_idx = user2idx[user_id]
    user_embedding = user_embeddings[user_idx].reshape(1, -1)  # Shape: [1, embedding_dim]
    user_bias = user_biases[user_idx]

    # Get items the user has already interacted with
    user_history = historical_data[historical_data['user_id'] == user_id]
    interacted_items = set(user_history['item_id'])
    interacted_item_indices = [item2idx[item_id] for item_id in interacted_items if item_id in item2idx]
    
    # List to store all non-interacted items with their predicted ratings
    all_candidates = []
    num_items = len(item2idx)
    # Iterate through all items
    for item_idx in range(num_items):
        # Skip items the user has already interacted with
        if item_idx in interacted_item_indices:
            continue
        item_id = idx_to_item[item_idx]
        item_embedding = item_embeddings[item_idx].reshape(1, -1)  # Shape: [1, embedding_dim]
        item_bias = item_biases[item_idx]
        # Compute predicted rating
        dot_product = np.dot(user_embedding, item_embedding.T)
        predicted_rating = global_bias + user_bias + item_bias + dot_product
        # Append the item and its predicted rating
        all_candidates.append((predicted_rating, item_id))

    # Use heapq.nlargest to get the top K candidates
    top_k_candidates = heapq.nlargest(top_k, all_candidates, key=lambda x: x[0])
    top_k_item_ids = [item_id for rating, item_id in top_k_candidates]
    

    # Re-rank using content-based filtering
    # Extract titles for items the user has interacted with
    user_interacted_items = item_metadata[item_metadata['item_id'].isin(interacted_items)]
    user_interacted_titles = user_interacted_items['title']
    
    if user_interacted_titles.empty:
        return {'error': f'No historical item titles found for user ID {user_id}'}
    
    # Compute TF-IDF vectors for user's historical items
    user_tfidf_matrix = tfidf_vectorizer.transform(user_interacted_titles)
    user_ratings = user_history['rating'].values
    # Compute weighted average to form user profile vector
    user_profile = np.average(user_tfidf_matrix.toarray(), axis=0, weights=user_ratings)
    # Normalize user profile vector
    if np.linalg.norm(user_profile) > 0:
        user_profile /= np.linalg.norm(user_profile)
    else:
        return {'error': f'User profile vector is zero for user ID {user_id}'}
    
    # Extract metadata for top_k candidate items
    top_k_metadata = item_metadata[item_metadata['item_id'].isin(top_k_item_ids)]
    if top_k_metadata.empty:
        return {'error': 'No metadata found for top_k candidate items'}
    
    top_k_titles = top_k_metadata['title']
    top_k_item_ids_in_metadata = top_k_metadata['item_id'].tolist()  ## only include items that have metadata
    
    # Compute TF-IDF vectors for top_k candidate items
    top_k_tfidf_matrix = tfidf_vectorizer.transform(top_k_titles)
    
    # Compute cosine similarity between user profile and top_k items
    similarities = cosine_similarity(user_profile.reshape(1, -1), top_k_tfidf_matrix).flatten()
    
    # Pair similarities with item IDs
    reranked_candidates = list(zip(similarities, top_k_item_ids_in_metadata))
    # Sort candidates by similarity in descending order
    reranked_candidates.sort(key=lambda x: -x[0])
    
    # Get top 5 items
    top_5_items = [item_id for similarity_score, item_id in reranked_candidates[:5]]
    
    return {'recommended_items': top_5_items}


if __name__ == '__main__':
    # For AWS SageMaker, host should be 0.0.0.0 and port 8080
    app.run(host='0.0.0.0', port=8080)
