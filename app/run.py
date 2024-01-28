import json
import plotly
import pandas as pd
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar     
#from sklearn.externals 
import joblib
from sqlalchemy import create_engine

app = Flask(__name__)


# tokenize function passed to TFIDF Vectorizer.
def tokenize(text):
    """ Basic NLP data processing:
    - case insensitive
    - punctuations removal
    - word tokenization
    - stop word removal
    - lemmatization
    """
    
    # make it case agnostic
    text = text.lower()
    # replace punctuations with spaces
    text = re.sub(r'[^[a-zA-Z0-9]]', " ", text)
    
    # get word tokens
    tokens = word_tokenize(text)
    
    # remove stop words
    words = [word for word in tokens if word not in stopwords.words('english')]

    # lemmatization
    lem = WordNetLemmatizer()
    words = [ lem.lemmatize(word=word) for word in words ]
    
    return words

# def tokenize(text):
#     """ Basic NLP data processing:
#     - word tokenization
#     - lemmatization
#     """
#     tokens = word_tokenize(text)
#     lemmatizer = WordNetLemmatizer()

#     clean_tokens = []
#     for tok in tokens:
#         clean_tok = lemmatizer.lemmatize(tok).lower().strip()
#         clean_tokens.append(clean_tok)

#     return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DevDB.db')
df = pd.read_sql_table('disaster_data_cleaned', engine)

# load model
model = joblib.load("../models/model_v3.pkl")
print(model)
print(dir(model))

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    """Rendering home page of the flask UI.

    Plots data distribution.
    """
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals

    # 1.distribution by genre
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)


    # 2. distribution for basic needs categories out of all categories
    basic_needs_pattern = 'food|water|clothing|shelter|medical|hospital' # use of regex pattern matching
    basic_needs = [col for col in df.columns if re.match(basic_needs_pattern, col.lower())!=None]

     # summing up without group since already binary encoded as 0 / 1.
    need_counts = [df[col].sum() for col in basic_needs]


    # create visuals

    graphs = [

    # 1. Genre counts (given)
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

    # 2. Basic necessities counts (added)
        {
            'data': [
                Bar(
                    x=basic_needs,
                    y=need_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Basic needs',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Basic Needs"
                }
            }
        },

    ]
    


    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """Event Handling function.
    This is run when user enters a message and submits. 
    It uses loaded ml model to generate predictios for the
    entered text and renders the predictions on the UI.
    """
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()