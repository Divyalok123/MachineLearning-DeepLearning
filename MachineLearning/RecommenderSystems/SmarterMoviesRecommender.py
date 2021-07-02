import pandas as pd

ratings = pd.read_csv('../../assets/ml-latest-small/ratings.csv', usecols=range(3), encoding='ISO-8859-1')
movies = pd.read_csv('../../assets/ml-latest-small/movies.csv', usecols=range(2), encoding='ISO-8859-1')
ratings = pd.merge(movies, ratings)

movieRatings = ratings.pivot_table(index=['userId'], columns=['title'], values='rating')
# print(movieRatings.head(10))

#correlation between every movie column
# correlationMatrix = movieRatings.corr() 
correlationMatrix = movieRatings.corr(method='pearson', min_periods=70)
# print(correlationMatrix.head(10))


#let's take the first user
user = movieRatings.loc[1].dropna()

similarMovies = pd.Series(dtype=object)

for i in range(0, len(user.index)):
    similars = correlationMatrix[user.index[i]].dropna()
    similars = similars.map(lambda x: x * user[i]) 
    similarMovies = similarMovies.append(similars)

similarMovies.sort_values(ascending=False, inplace=True)
# print(similarMovies.head(10))

similarMovies = similarMovies.groupby(similarMovies.index).sum()
similarMovies.sort_values(inplace=True, ascending=False)
print(similarMovies.head(10))

for i in range(0, len(user.index)):
    if(similarMovies[user.index[i]]): similarMovies = similarMovies.drop(user.index[i], inplace=True)
print(similarMovies.head(10))