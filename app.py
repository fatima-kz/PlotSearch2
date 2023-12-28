from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer as tvec
from sklearn.metrics.pairwise import cosine_similarity as sim
#import numpy as np
from collections import deque
#from transformers import pipeline
#import scipy.sparse

app = Flask(__name__)
queue = deque(maxlen=4)
#fill_mask = pipeline("fill-mask", model="bert-base-uncased")
dataFrame = pd.read_excel('D:\horror2.xlsx', sheet_name='horror')
movieNames = dataFrame['name'].tolist()
movieRating = dataFrame['rating'].tolist()
descriptions = dataFrame['description'].tolist()
newDescriptions = []
newMovies = []
newRating = []
for obj1, obj2, obj in zip(movieNames,movieRating,descriptions):
    if pd.notna(obj) and pd.notna(obj2):
        newDescriptions.append(str(obj))
        newMovies.append(str(obj1))
        newRating.append(str(obj2))
    #else:
    #   newDescriptions.append('')
vectorizer = tvec(stop_words='english', max_features=3000)
matrix = vectorizer.fit_transform(newDescriptions)
@app.route('/', methods= ['POST','GET'])

def index():
    if request.method == 'POST':
        givenPlot = request.form['plot']
        resultMovies = similaritySearch(givenPlot)
        addToHistory(resultMovies)
        return render_template('index.html', movies=resultMovies, histories=queue)
    else:
        return render_template('index.html')

def similaritySearch(newPlot):
    #sparse_matrix = scipy.sparse.csr_matrix(matrix)
    #vector = matrix.toarray()
    vectorize = vectorizer.transform([newPlot])
    similaritySearching = sim(vectorize, matrix)[0]
    simMoviesList = []
    for title, rating, similarity in zip(newMovies, newRating, similaritySearching):
        simMovies = {"title": title, "rating": rating, "similarity": similarity*100}
        simMoviesList.append(simMovies)
    simMoviesList.sort(key=lambda x: x['similarity'], reverse=True)
    return simMoviesList[:5]

def addToHistory(similarMovies):
    queue.append(similarMovies)

if __name__ == "__main__":
    app.run(debug=True)

