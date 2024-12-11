# test.py

import unittest
import pandas as pd
import os
from recommend import recommend

class TestRecommendationSystem(unittest.TestCase):

    def setUp(self):
        """Set up test data and paths."""
        self.train_data_path = 'data_processed/train_data.csv'
        self.test_data_path = 'data_processed/test_data.csv'
        self.content_similarity_path = 'model/content_similarity.csv'
        self.movies_path = 'data/movies.csv'

        # Ensure files exist
        self.assertTrue(os.path.exists(self.train_data_path), "Train data file missing.")
        self.assertTrue(os.path.exists(self.test_data_path), "Test data file missing.")
        self.assertTrue(os.path.exists(self.content_similarity_path), "Content similarity file missing.")
        self.assertTrue(os.path.exists(self.movies_path), "Movies file missing.")

        self.test_data = pd.read_csv(self.test_data_path)
        self.train_data = pd.read_csv(self.train_data_path)

    def test_recommendation_output(self):
        """Test that recommendations are generated for a random user."""
        random_user_id = self.test_data['userId'].sample(1).iloc[0]
        recommendations = recommend(
            user_id=random_user_id,
            train_data_path=self.train_data_path,
            test_data_path=self.test_data_path,
            content_similarity_path=self.content_similarity_path,
            num_recommendations=5
        )
        self.assertIsInstance(recommendations, pd.DataFrame)
        self.assertTrue(len(recommendations) > 0, "No recommendations generated.")

    def test_content_similarity_file(self):
        """Test that the content similarity file is valid."""
        content_similarity = pd.read_csv(self.content_similarity_path, index_col=0)
        self.assertIsInstance(content_similarity, pd.DataFrame)
        self.assertTrue(not content_similarity.empty, "Content similarity matrix is empty.")

if __name__ == "__main__":
    unittest.main()
