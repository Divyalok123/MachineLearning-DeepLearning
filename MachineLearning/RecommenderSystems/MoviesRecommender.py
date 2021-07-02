import pandas as pd

ratings = pd.read_csv('../../assets/ml-latest-small/ratings.csv', usecols=range(3), encoding='ISO-8859-1')
movies = pd.read_csv('../../assets/ml-latest-small/movies.csv', usecols=range(2), encoding='ISO-8859-1')

# print(ratings.head())
# print(movies.head())

ratings = pd.merge(movies, ratings)
# print(ratings.tail(10))

movieRatings = ratings.pivot_table(index=['userId'], columns=['title'], values='rating')
# print(movieRatings.tail(10))

currMovieRatings = movieRatings['GoldenEye (1995)']
currMovieRatings = currMovieRatings.dropna() #drop NAN values
# print(currMovieRatings.head(10))

similarMovies = movieRatings.corrwith(currMovieRatings) #getting correlation with other movies
similarMovies = similarMovies.dropna()

df = pd.DataFrame(similarMovies)
# print(df.head(10))

#sort the results with similarity score
similarMovies = similarMovies.sort_values(ascending=False)
print(similarMovies.head(20))

#how many ratings exits for each movie
import numpy as np
movieStats = ratings.groupby('title').agg({'rating': [np.size, np.mean]})

#remove movies which are only rated by very few people
popularMovies = movieStats['rating']['size'] >= 120
# print(popularMovies.head(10))

print(movieStats[popularMovies])

sortedPopular = movieStats[popularMovies].sort_values([('rating', 'mean')], ascending=False)
# print(sortedPopular.head(10))

similarmoviesdf = pd.DataFrame(similarMovies, columns=['similarity'])
print(similarmoviesdf)

df = movieStats[popularMovies].join(similarmoviesdf)
df = df.sort_values(['similarity'], ascending=False)
print(df.head(10))

