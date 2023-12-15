from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer as tvec
from sklearn.metrics.pairwise import cosine_similarity as sim
import numpy as np
from collections import deque
import scipy.sparse

app = Flask(__name__)
queue = deque(maxlen=4)
@app.route('/', methods= ['POST','GET'])

def index():
    if request.method == 'POST':
        givenPlot = request.form['plot']
        resultMovies = similaritySearch(givenPlot)
        addToHistory(resultMovies)
        return render_template('index.html', movies=resultMovies, histories=queue)
    else:
        return render_template('index.html')

def similaritySearch(new_plot):
    dataFrame = pd.read_excel('D:\horror2.xlsx', sheet_name='horror')
    movieNames = dataFrame['name'].tolist()
    descriptions = dataFrame['description'].tolist()
    newDescriptions = []
    for obj in descriptions:
        if pd.notna(obj):
            newDescriptions.append(str(obj))
        else:
            newDescriptions.append('')
    vectorizer = tvec(stop_words='english', max_features=3000)
    matrix = vectorizer.fit_transform(newDescriptions)
    sparse_matrix = scipy.sparse.csr_matrix(matrix)
    #vector = matrix.toarray()
    vectorize = vectorizer.transform([new_plot]).toarray()
    similaritySearching = sim(vectorize, sparse_matrix)[0]
    simMoviesList = []
    for title, similarity in zip(movieNames, similaritySearching):
        simMovies = {"title": title, "similarity": similarity}
        simMoviesList.append(simMovies)
    simMoviesList.sort(key=lambda x: x['similarity'], reverse=True)
    return simMoviesList[:5]

def addToHistory(similarMovies):
    queue.append(similarMovies)

if __name__ == "__main__":
    app.run(debug=True)

