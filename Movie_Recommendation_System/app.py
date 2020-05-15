import numpy as np
from flask import Flask, request, jsonify, render_template,json
import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
df = pd.read_csv("movie_dataset.csv")
vote_counts=df['vote_count']

features = ['keywords','cast','genres','director']
def combine_features(row):
    return row['keywords'] +" "+row['cast']+" "+row["genres"]+" "+3*row["director"]
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


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUImovie_user_likes
    '''
    print(request)
    print(request.json)
    lst=[]
    movie_user_likes = request.json["Movie"]
    print(movie_user_likes)
    movie_index = get_index_from_title(movie_user_likes)
    similar_movies = list(enumerate(cosine_sim[movie_index]))
    sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:]
    i = 0
    print("Top 5 similar movies to " + movie_user_likes + " are:\n")
    for element in sorted_similar_movies:
        if(vote_counts[element[0]]>=3000):
            lst.append(get_title_from_index(element[0]))
            i = i + 1
            if i >= 5:
                break
    print(lst)
    return json.dumps({'lst':lst})


if __name__ == "__main__":
    app.run(debug=True)