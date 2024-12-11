import pandas as pd
import numpy as np
import os

def load_data():
    """Load data from CSV files."""
    ratings = pd.read_csv('data/ratings.csv')
    movies = pd.read_csv('data/movies.csv')
    tags = pd.read_csv('data/tags.csv')
    links = pd.read_csv('data/links.csv')
    return ratings, movies, tags, links

def filter_data(ratings):
    """Filter data to remove users with fewer than 10 reviews."""
    # Count the number of ratings per user
    user_review_counts = ratings['userId'].value_counts()

    # Identify users with at least 10 reviews
    active_users = user_review_counts[user_review_counts >= 10].index

    # Filter ratings
    filtered_ratings = ratings[ratings['userId'].isin(active_users)]
    return filtered_ratings

def create_genres_matrix(movies):
    """Create a genres matrix from the movies dataset."""
    genres_matrix = movies[['movieId']].copy()
    genres_dummies = movies['genres'].str.get_dummies(sep='|')
    genres_matrix = genres_matrix.join(genres_dummies).set_index('movieId')
    return genres_matrix

def create_tags_data(tags, movies):
    """Merge tags data with movies to create a tags dataset with movie IDs."""
    movie_tags = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
    merged_tags = movies[['movieId']].merge(movie_tags, on='movieId', how='left')
    merged_tags['tag'] = merged_tags['tag'].fillna('')
    return merged_tags

def split_data(filtered_ratings, test_size=0.2, random_state=42):
    """Split the filtered ratings data into training and testing sets."""
    user_ids = filtered_ratings['userId'].unique()
    np.random.seed(random_state)
    np.random.shuffle(user_ids)

    split_point = int(len(user_ids) * (1 - test_size))
    train_users = user_ids[:split_point]
    test_users = user_ids[split_point:]

    train_data = filtered_ratings[filtered_ratings['userId'].isin(train_users)]
    test_data = filtered_ratings[filtered_ratings['userId'].isin(test_users)]

    return train_data, test_data

def save_preprocessed_data(filtered_ratings, train_data, test_data, genres_matrix, tags_data, output_dir='data_processed'):
    """Save preprocessed data to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    filtered_ratings.to_csv(os.path.join(output_dir, 'filtered_ratings.csv'), index=False)
    train_data.to_csv(os.path.join(output_dir, 'train_data.csv'), index=False)
    test_data.to_csv(os.path.join(output_dir, 'test_data.csv'), index=False)
    genres_matrix.to_csv(os.path.join(output_dir, 'genres_matrix.csv'))
    tags_data.to_csv(os.path.join(output_dir, 'movies_with_tags.csv'), index=False)
    print(f"Preprocessed data saved to '{output_dir}'.")

if __name__ == "__main__":
    # Load data
    ratings, movies, tags, links = load_data()

    # Filter ratings
    filtered_ratings = filter_data(ratings)

    # Create genres matrix
    genres_matrix = create_genres_matrix(movies)

    # Create tags data
    tags_data = create_tags_data(tags, movies)

    # Split data
    train_data, test_data = split_data(filtered_ratings)

    # Save preprocessed data
    save_preprocessed_data(filtered_ratings, train_data, test_data, genres_matrix, tags_data)
