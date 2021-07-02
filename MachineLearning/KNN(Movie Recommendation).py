from collections import defaultdict
import pandas as pd
import numpy as np
from scipy import spatial

ratings = pd.read_csv('../assets/ml-latest-small/ratings.csv', usecols=range(3))
movieProperties = ratings.groupby('movieId').agg({'rating': [np.size, np.mean]})

movieNumRatings = pd.DataFrame(movieProperties['rating']['size'])
movieNormalizedNumRatings = movieNumRatings.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

movies = pd.read_csv('../assets/ml-latest-small/movies.csv')
genrelist = []

for i in range(0, len(movies.index)):
    genres = movies.loc[i]['genres'].split('|')
    for j in genres:
        genrelist.append(j)

allgenres = list(set(genrelist))
allgenres.remove('(no genres listed)')
# print(allgenres)

numgenres = len(allgenres)
# print(numgenres)

indice = 0
genredict = defaultdict()

for genre in allgenres:
    genredict[genre] = indice
    indice += 1

movieDict = {}
for i in range(len(movies.index)):
    movie = movies.loc[i]
    movieID = movie['movieId']
    name = movie['title']
    genres = movie['genres'].split('|')

    if(genres[0] != '(no genres listed)'):
        genrehash = [0]*numgenres
        for genre in genres:
            genrehash[genredict[genre]] = 1

    genres = map(int, genrehash)
    # print(genres)
    try:
        movieDict[movieID] = (name, np.array(list(genres)), movieNormalizedNumRatings.loc[movieID].get('size'), movieProperties.loc[movieID].rating.get('mean'))
    except:
        pass

# print(movieDict)

def ComputeDistance(a, b):
    genresA = a[1]
    genresB = b[1]
    genreDistance = spatial.distance.cosine(genresA, genresB)
    popularityA = a[2]
    popularityB = b[2]
    popularityDistance = abs(popularityA - popularityB)
    return genreDistance + popularityDistance
    
ComputeDistance(movieDict[2], movieDict[4])


import operator
def getNeighbors(movieID, K):
    distances = []
    for movie in movieDict:
        if (movie != movieID):
            dist = ComputeDistance(movieDict[movieID], movieDict[movie])
            distances.append((movie, dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(K):
        neighbors.append(distances[x][0])
    return neighbors

print('movie with id 1: ', movieDict[2])

K = 7
avgRating = 0
neighbors = getNeighbors(2, K)
for neighbor in neighbors:
    avgRating += movieDict[neighbor][3]
    print (movieDict[neighbor][0] + " " + str(movieDict[neighbor][3]))
    
avgRating /= K

print(avgRating)