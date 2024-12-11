# visualize.py

import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

ratings = pd.read_csv('data/ratings.csv')
movies = pd.read_csv('data/movies.csv')
tags = pd.read_csv('data/tags.csv')
links = pd.read_csv('data/links.csv')
movies_genres_expanded = pd.concat([movies[['movieId']], movies['genres'].str.get_dummies(sep='|')], axis=1)
merged_data = ratings.merge(movies_genres_expanded, on='movieId')


# Similarity matrix
def generate_similarity_matrix(genres_path, tags_path):
    """
    Generate a content-based similarity matrix dynamically.
    
    Args:
    - genres_path: Path to the genres CSV.
    - tags_path: Path to the tags CSV.
    
    Returns:
    - similarity_df: DataFrame containing the similarity matrix.
    """
    # Load genres and tags
    genres = pd.read_csv(genres_path).set_index('movieId')
    tags = pd.read_csv(tags_path).set_index('movieId')['tag'].fillna('')

    # Compute TF-IDF for tags
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(tags)

    # Combine genres and tags
    genre_matrix = sp.csr_matrix(genres.values)
    content_matrix = sp.hstack([genre_matrix, tfidf_matrix], format='csr')

    # Compute similarity matrix
    similarity_matrix = cosine_similarity(content_matrix)
    similarity_df = pd.DataFrame(
        similarity_matrix, index=genres.index, columns=genres.index
    )
    return similarity_df

def plot_similarity_matrix(similarity_df, output_path='graph/similarity_matrix.png', top_n=50):
    """
    Plot and save a heatmap of the similarity matrix.
    
    Args:
    - similarity_df: DataFrame containing the similarity matrix.
    - output_path: Path to save the similarity heatmap.
    - top_n: Number of movies to include in the plot (for performance reasons).
    """
    # Select top N movies for plotting
    subset = similarity_df.iloc[:top_n, :top_n]

    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(subset, cmap='coolwarm', xticklabels=False, yticklabels=False)
    plt.title(f"Similarity Matrix (Top {top_n} Movies)")
    plt.xlabel("Movies")
    plt.ylabel("Movies")

    # Save heatmap
    plt.savefig(output_path)
    plt.close()
    print(f"Similarity matrix heatmap saved to {output_path}")

### EXPLORATORY ANALYSIS GRAPHS

def ratings_graph():
    plt.figure(figsize=(8, 6))
    ratings['rating'].value_counts().sort_index().plot(kind='bar')
    plt.title("Distribution of Movie Ratings")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.savefig('graph/rating_dist.png')

def top_ratings():
    top_movies = ratings['movieId'].value_counts().head(20)
    top_movies_titles = movies[movies['movieId'].isin(top_movies.index)]

    plt.figure(figsize=(10, 6))
    plt.barh(top_movies_titles['title'], top_movies.values, color='skyblue')
    plt.title("Top 20 Most Rated Movies")
    plt.xlabel("Number of Ratings")
    plt.gca().invert_yaxis()
    plt.savefig('graph/top_ratings.png')

def genre_ratings():
    genre_avg_ratings = merged_data.drop(columns=['userId', 'timestamp', 'rating']).multiply(merged_data['rating'], axis="index").sum() / movies_genres_expanded.sum()

    # Plotting the results
    plt.figure(figsize=(12, 6))
    genre_avg_ratings.sort_values(ascending=False).plot(kind='bar', color='salmon')
    plt.title("Average Rating per Genre")
    plt.xlabel("Genre")
    plt.ylabel("Average Rating")
    plt.savefig('graph/genre_ratings.png')

def genre_freq():
    # Split genres into individual columns and sum occurrences
    genre_counts = movies['genres'].str.get_dummies(sep='|').sum().sort_values(ascending=False)

    # Plot the genre frequency
    plt.figure(figsize=(12, 6))
    genre_counts.plot(kind='bar', color='skyblue')
    plt.title("Frequency of Each Genre")
    plt.xlabel("Genre")
    plt.ylabel("Count")
    plt.savefig('graph/genre_frequency.png')


genres_path = 'data_processed/genres_matrix.csv'
tags_path = 'data_processed/movies_with_tags.csv'
s = generate_similarity_matrix(genres_path, tags_path)

plot_similarity_matrix(s)
ratings_graph()
top_ratings()
genre_ratings()
genre_freq()