import pandas as pd
import numpy as np
import os


def recommend(user_id, train_data_path, test_data_path, content_similarity_path, output_dir="recommendations", num_recommendations=5):
    """
    Recommend movies for a user using a precomputed content similarity matrix and save results.
    
    Args:
    - user_id: ID of the target user.
    - train_data_path: Path to the training data CSV file.
    - test_data_path: Path to the test data CSV file.
    - content_similarity_path: Path to the precomputed content similarity matrix.
    - output_dir: Directory to save individual recommendations.
    - num_recommendations: Number of recommendations to return.
    
    Returns:
    - DataFrame with recommended movie titles and scores.
    """
    # Load data
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    content_similarity = pd.read_csv(content_similarity_path, index_col=0)

    # Combine train and test data for filtering purposes
    all_ratings = pd.concat([train_data, test_data], ignore_index=True)

    # Get the movies the user has rated positively
    user_ratings = test_data[(test_data['userId'] == user_id) & (test_data['rating'] >= 3.0)]
    user_movies = user_ratings['movieId'].values

    # Collect similarity scores for each movie the user liked
    similar_movies_list = []
    for movie_id in user_movies:
        if str(movie_id) in content_similarity.columns:
            similar_movies_list.append(content_similarity[str(movie_id)])
    
    if not similar_movies_list:
        print(f"No recommendations available for User {user_id}.")
        return pd.DataFrame(columns=['title', 'score'])

    # Calculate similarity scores and remove movies the user has already rated
    similar_movies = pd.concat(similar_movies_list, axis=1).mean(axis=1)
    all_user_rated_movies = all_ratings[all_ratings['userId'] == user_id]['movieId']
    similar_movies = similar_movies.drop(labels=all_user_rated_movies, errors='ignore')

    # Get top recommendations
    recommendations = similar_movies.sort_values(ascending=False).head(num_recommendations)

    # Map movie IDs to titles
    movies = pd.read_csv('data/movies.csv').set_index('movieId')
    recommended_movies = movies.loc[recommendations.index]
    recommended_movies['score'] = recommendations.values

    # Save recommendations to a CSV file
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"user_{user_id}_recommendations.csv")
    recommended_movies[['title', 'score']].to_csv(output_file, index=False)
    print(f"Recommendations for User {user_id} saved to {output_file}")

    return recommended_movies[['title', 'score']]


if __name__ == "__main__":
    # File paths
    train_data_path = 'data_processed/train_data.csv'
    test_data_path = 'data_processed/test_data.csv'
    content_similarity_path = 'model/content_similarity.csv'
    movie_titles_path = 'data/movies.csv'
    output_dir = 'recommendations'

    # Load test data
    test_data = pd.read_csv(test_data_path)

    # Select a random user from the test dataset
    random_user_id = np.random.choice(test_data['userId'].unique())

    # Generate recommendations for the user
    recommendations = recommend(
        user_id=random_user_id,
        train_data_path=train_data_path,
        test_data_path=test_data_path,
        content_similarity_path=content_similarity_path,
        output_dir=output_dir,
    )

    # Display the user's input movies and ratings
    user_ratings = test_data[test_data['userId'] == random_user_id]
    movies = pd.read_csv(movie_titles_path).set_index('movieId')
    user_movies = user_ratings.merge(movies, left_on='movieId', right_index=True)

    print(f"User {random_user_id}'s Rated Movies and Ratings:")
    print(user_movies[['title', 'rating']])

    print(f"Recommended movies for User {random_user_id}:")
    print(recommendations)

