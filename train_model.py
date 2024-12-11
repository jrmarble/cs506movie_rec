import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

def train_content_similarity(train_data_path, genres_path, tags_path, output_path, alpha=0.7, beta=0.8):
    """
    Train a content-based similarity matrix considering movie genres, tags, and popularity.
    
    Args:
    - train_data_path: Path to the training data CSV.
    - genres_path: Path to preprocessed genres CSV.
    - tags_path: Path to preprocessed tags CSV.
    - output_path: Path to save the similarity matrix.
    - alpha: Weight for combining number of ratings and average rating into popularity score.
    - beta: Weight for combining content similarity and popularity.
    """
    # Load training data
    train_data = pd.read_csv(train_data_path)
    train_movie_ids = train_data['movieId'].unique()

    # Load genres and tags
    genres = pd.read_csv(genres_path).set_index('movieId')
    genres = genres.loc[genres.index.isin(train_movie_ids)]
    tags = pd.read_csv(tags_path).set_index('movieId')
    tags = tags.loc[tags.index.isin(train_movie_ids), 'tag'].fillna('')

    # Compute popularity metrics
    movie_stats = train_data.groupby('movieId').agg(
        num_ratings=('rating', 'count'),
        avg_rating=('rating', 'mean')
    )
    movie_stats['popularity'] = (
        alpha * movie_stats['num_ratings'] / movie_stats['num_ratings'].max() +
        (1 - alpha) * movie_stats['avg_rating'] / 5.0  # Normalize avg_rating to [0, 1]
    )
    movie_stats = movie_stats.loc[genres.index]  # Align with genres index

    # Compute TF-IDF for tags
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(tags)

    # Combine genres and tags
    genre_matrix = sp.csr_matrix(genres.values)
    content_matrix = sp.hstack([genre_matrix, tfidf_matrix], format='csr')

    # Compute content similarity
    content_similarity = cosine_similarity(content_matrix)

    # Save the similarity matrix
    similarity_df = pd.DataFrame(content_similarity, index=genres.index, columns=genres.index)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    similarity_df.to_csv(output_path)
    print(f"Content similarity matrix with popularity saved to {output_path}")

if __name__ == "__main__":
    train_data_path = 'data_processed/train_data.csv'
    genres_path = 'data_processed/genres_matrix.csv'
    tags_path = 'data_processed/movies_with_tags.csv'
    output_path = 'model/content_similarity.csv'

    # Train and save the similarity matrix
    train_content_similarity(train_data_path, genres_path, tags_path, output_path)
