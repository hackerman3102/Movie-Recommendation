import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
df = pd.read_csv("movie_dataset.csv")
vote_counts=df['vote_count']
features = ['keywords','cast','genres','director']
def combine_features(row):
    return 1*row['keywords'] + " " + row['cast'] + " " + row["genres"] + " " + 3*row["director"]
for feature in features:
    df[feature] = df[feature].fillna('')
df["combined_features"] = df.apply(combine_features,axis=1)
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])
cosine_sim = cosine_similarity(count_matrix)
def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]
def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]
movie_user_likes = "Inside Out "
movie_index = get_index_from_title(movie_user_likes)
similar_movies =  list(enumerate(cosine_sim[movie_index]))
sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)[1:]
i=0
lst=[]
print("Top 5 similar movies to "+movie_user_likes+" are:\n")
for element in sorted_similar_movies:
    if (vote_counts[element[0]] >= 3000):
        print(get_title_from_index(element[0]))
        print(element)


        lst.append(get_title_from_index(element[0]))

        i = i + 1
        if i >= 10:
            break