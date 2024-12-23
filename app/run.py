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
import logging 

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("run.log"),
        logging.StreamHandler()
    ]
)


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
        # Load data
        database_filepath = 'data/DisasterResponse.db'
        engine = create_engine(f'sqlite:///{database_filepath}')
        df = pd.read_sql_table('DisasterMessages', engine)
        logging.info("Data loaded successfully.")

        # Load model
        model_filepath = 'models/classifier.pkl'
        model = pickle.load(open(model_filepath, 'rb'))
        logging.info("Model loaded successfully.")

    except Exception as e:
        logging.error(f"Error loading resources: {e}")
        sys.exit(1)

@app.route('/')
@app.route('/index')
def index():
    # Data for visualizations
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    category_counts = df.iloc[:, 4:].sum().sort_values(ascending=False)
    category_names = category_counts.index

    message_lengths = df['message'].str.len()

    # Calculate correlation matrix for categories
    correlation_matrix = df.iloc[:, 4:].corr()

    # Create graphs
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
                    x=category_names[:10], # Top 10 categories
                    y=category_counts[:10]
                )
            ],
            'layout': {
                'title': 'Top 10 Message Categories',
                'yaxis': {'title': 'Count'},
                'xaxis': {'title': 'Category'}
            }
        },
        {
            'data': [
                Bar(
                    x=['Short', 'Medium', 'Long'],
                    y=[
                        sum(message_lengths <= 50),
                        sum((message_lengths > 50) & (message_lengths <= 200)),
                        sum(message_lengths > 200)
                    ]
                )
            ],
            'layout': {
                'title': 'Message Length Distribution',
                'yaxis': {'title': 'Count'},
                'xaxis': {'title': 'Length Category'}
            }
        },
        # Correlation heatmap for categories
        {
            'data': [
                {
                    'z': correlation_matrix.values.tolist(),
                    'x': correlation_matrix.columns.tolist(),
                    'y': correlation_matrix.columns.tolist(),
                    'type': 'heatmap',
                    'colorscale': 'Viridis'
                }
            ],
            'layout': {
                'title': 'Correlation Heatmap of Categories',
                'yaxis': {'title': 'Categories'},
                'xaxis': {'title': 'Categories'}
            }
        }

    ]
    # Encode graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('master.html', ids=ids, graphJSON=graphJSON)

@app.route('/go')
def go():
    # Get user input
    query = request.args.get('query', '').strip()

    if not query:
        logging.warning("Empty query received.")
        return render_template('error.html', error_message="Please provide a valid input message.")
    
    # Predict categories
    try:
        classification_labels = model.predict([query])[0]
        classification_results = dict(zip(df.columns[4:], classification_labels))

        logging.info(f"User input: {query}")
        logging.info(f"Classification results: {classification_results}")

        return render_template(
            'go.html',
            query=query,
            classification_result=classification_results
        )
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return render_template('error.html', error_message="Unable to classify the message.")
    
def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()
