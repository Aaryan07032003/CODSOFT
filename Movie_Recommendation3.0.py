import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Movie data
movies = pd.DataFrame({
    'title': ['Pathaan', 'Brahmastra', 'Animal', 'Gehraiyaan', 'Sooryavanshi', 'Bell Bottom', 'Shershaah', 'Maidaan'],
    'genre': ['Action|Thriller', 'Fantasy|Adventure', 'Crime|Drama', 'Romance|Drama', 'Action|Drama', 'Thriller|Spy', 'Biographical|War', 'Sports|Drama']
})

# Movie and its genre
user_movie = input("Enter a movie: ")
user_genre = input("Enter the genre of the movie in the form Genre1|Genre2(Eg:Comedy|Drama): ")

# Check if the user-provided movie is already in the list
if user_movie not in movies['title'].values:
    # Add user-provided movie to the movie data
    new_movie = pd.DataFrame({'title': [user_movie], 'genre': [user_genre]})
    movies = pd.concat([movies, new_movie], ignore_index=True)

# Converting genres to TF-IDF vectors
tfidf = TfidfVectorizer()
genre_matrix = tfidf.fit_transform(movies['genre'])

# Similarity scores
cosine_sim = linear_kernel(genre_matrix, genre_matrix)

# Movie Recommendations
def get_recommendations(title, cosine_sim=cosine_sim, movies=movies):
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:5] 
    movie_indices = [i[0] for i in sim_scores]
    recommended_movies = movies['title'].iloc[movie_indices].values
    return [movie for movie in recommended_movies if movie != title]  

recommendations = get_recommendations(user_movie)
print("Recommended movies based on your input:")
for movie in recommendations:
    print(movie)
