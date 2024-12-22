import sys
import json
import pandas as pd
import plotly
from flask import Flask, render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pickle
import os


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]
    return clean_tokens

@app.before_request
def load_resources():
    """
    Load resources (data and model) when the application starts.
    """
    global df, model
    try:
        import os

        # Construct absolute paths
        database_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'DisasterResponse.db'))
        model_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'classifier.pkl'))

        # Debugging: Print the resolved paths
        print(f"Resolved database path: {database_filepath}")
        print(f"Resolved model path: {model_filepath}")

        # Load the database
        engine = create_engine(f'sqlite:///{database_filepath}')
        df = pd.read_sql_table('DisasterMessages', engine)

        # Load the model
        model = pickle.load(open(model_filepath, 'rb'))

    except Exception as e:
        print(f"Error loading resources: {e}")
        sys.exit(1)

@app.route('/')
@app.route('/index')
def index():
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    category_counts = df.iloc[:, 4:].sum().sort_values(ascending=False)
    category_names = category_counts.index

    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {'title': 'Count'},
                'xaxis': {'title': 'Genre'}
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Categories',
                'yaxis': {'title': 'Count'},
                'xaxis': {'title': 'Category'}
            }
        }
    ]

    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('master.html', ids=ids, graphJSON=graphJSON)

@app.route('/go')
def go():
    query = request.args.get('query', '')
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()
