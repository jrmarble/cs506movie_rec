# Movie Recommendation System

## Project Overview
This project aims to create a machine learning-powered movie recommendation system that suggests movies to users based on their viewing preferences. The system will analyze user interactions, such as ratings, along with movie metadata to provide personalized recommendations.

## Goals
- Develop a recommendation system that predicts user movie preferences based on past interactions.
- Compare different recommendation techniques such as collaborative filtering, content-based filtering, and hybrid models.
- Measure performance using metrics like Mean Squared Error (MSE), Precision, and Recall.

## Data Collection
The data is be sourced from the [MovieLens](https://grouplens.org/datasets/movielens/) small dataset. This dataset includes:
- User ratings and interactions.
- Movie metadata (e.g., genre, release year).
- User demographic information (age, gender, etc.).

The data contains 100,000 ratings and 3,600 tag applications applied to 9,000 movies by 600 users.

## Data Preprocessing and Feature Engineering
Several data preprocessing steps were taken to improve data quality and reduce noise:
- **Filtering Active Users**: Removed users who had fewer than 10 reviews to minimize noise.
- **One-Hot Encoding Genres**: Transformed genre information into a binary format using one-hot encoding, allowing us to use genres as features in our content-based model.
- **Tag Processing with TF-IDF**: Used TF-IDF vectorization to convert tags into weighted features, helping to capture unique descriptive terms for each movie.


## Data Visualization
We used various visualizations to explore the dataset and gain insights:
- **Rating Distribution**: Bar chart showing the distribution of ratings, indicating common rating biases.
- **Most Rated Movies**: Bar chart of the top 20 most-rated movies, highlighting popular titles.
- **Average Rating per Genre**: Bar chart showing the average user rating for each genre, providing insights into user preferences.

## Modeling Approach
The project currently explores the following recommendation techniques:

### 1. Content-Based Filtering
Using movie genres and user-provided tags as features, this approach recommends movies based on the similarity of content features:
- **Genres**: Encoded using one-hot encoding to allow binary comparisons.
- **Tags**: Processed with TF-IDF to assign weights to frequently used terms.
- **Similarity Computation**: Used cosine similarity between movies to identify similar titles for recommendation.

### 2. Collaborative Filtering (To Be Implemented)
We plan to implement collaborative filtering to leverage patterns in user behavior and ratings, aiming to recommend movies based on what similar users have rated highly.

### 3. Hybrid Models (Future Direction)
Once content-based and collaborative filtering models are functional, we aim to combine them into a hybrid model to enhance recommendation accuracy.


### Future Techniques
- **Matrix Factorization (SVD)**: We may explore matrix factorization to reduce dimensionality and uncover latent factors in user-movie interactions.
- **Neural Collaborative Filtering**: Potential future work to leverage deep learning for more complex interactions.

## Preliminary Results
- **Content-Based Filtering**: Initial recommendations from the content-based model are promising, especially for users with strong genre preferences. Recommendations are based on genre and tag similarity, which aligns with user interests in specific themes or topics.
- **Metrics**: We evaluated initial content-based recommendations using Mean Squared Error (MSE) as an indicator, though further evaluation with Precision and Recall will be performed after implementing collaborative and hybrid models.

## Test Plan
- **Data Split**: The dataset is split into training (80%) and testing (20%) sets to evaluate model performance.
- **Evaluation Metrics**: Current models are evaluated using MSE. For content-based recommendations, further qualitative evaluation will be conducted by assessing genre and tag alignment with user interests.
- **Cross-Validation**: Cross-validation will be used to validate the robustness of future models, ensuring they generalize well across different user subsets.

## Timeline
- **Midterm Completion**: All data preprocessing, visualization, and initial content-based filtering methods are complete.
- **Future Steps**: The next phase will involve implementing collaborative filtering, refining content-based filtering, and testing hybrid approaches. I aim to complete testing and evaluation of all methods by the final deadline.

## Midterm demo link
[YouTube Presentation Link](https://www.youtube.com/watch?v=RM6n39x6xuA)  
