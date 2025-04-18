import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommender:
    def __init__(self, ratings_path, tags_path, movies_path):
        self.ratings_path = ratings_path
        self.tags_path = tags_path
        self.movies_path = movies_path
        self.movies_content = None
        self.similarity_matrix = None

    def load_and_process_data(self):
        Ratings = pd.read_csv(self.ratings_path)
        Tags = pd.read_csv(self.tags_path)
        Movies = pd.read_csv(self.movies_path)

        movies = Movies.drop_duplicates("title")
        movies_with_ratings = Ratings.merge(movies, on="movieId")
        movies_with_ratings.drop(columns=["timestamp"], inplace=True)
        movie_matrix = movies_with_ratings.merge(Tags, on=["userId", "movieId"])

        movie_matrix = movie_matrix.groupby(
            ['userId', 'movieId', 'rating', 'title', 'genres'], as_index=False
        ).agg({'tag': lambda tags: '|'.join(set(tags.dropna()))})

        avg_ratings = movie_matrix.groupby("movieId")["rating"].mean()
        high_rated_movies = avg_ratings[avg_ratings >= 3.5].index
        filtered_movies = movie_matrix[movie_matrix["movieId"].isin(high_rated_movies)].copy()
        filtered_movies["features"] = filtered_movies["tag"]

        self.movies_content = filtered_movies.groupby(
            ["movieId", "title", "genres"], as_index=False
        ).agg({"features": lambda x: ' '.join(set(x.dropna()))})

        vectorizer = CountVectorizer(tokenizer=lambda x: x.split("|"))
        feature_matrix = vectorizer.fit_transform(self.movies_content["features"])
        self.similarity_matrix = cosine_similarity(feature_matrix)

    def get_recommendations(self, selected_title, top_n=5):
        if selected_title not in self.movies_content['title'].values:
            raise ValueError("Selected title not found")

        index = self.movies_content[self.movies_content['title'] == selected_title].index[0]
        similarity_scores = list(enumerate(self.similarity_matrix[index]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

        recommended_titles = [self.movies_content.iloc[i[0]]['title'] for i in similarity_scores]
        return recommended_titles
