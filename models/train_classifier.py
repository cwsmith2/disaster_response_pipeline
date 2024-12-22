import sys
import pandas as pd
import pickle
from sklearn import pipeline
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download(['punkt', 'wordnet','punkt_tab'])

def load_data(database_filepath):
    """
    Load data from SQLite database.
    
    Args:
        database_filepath: string. Filepath for SQLite database containing cleaned data.
        
    Returns:
        X: dataframe. Features dataframe.
        Y: dataframe. Labels dataframe.
        category_names: list. List of category names for classification.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('DisasterMessages', engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    
    # Find problematic columns
    non_binary_columns = Y.columns[~Y.applymap(lambda x: x in [0, 1]).all()]
    print(f"Non-binary values found in columns: {non_binary_columns.tolist()}")
    
    # Raise error if non-binary values exist
    if not non_binary_columns.empty:
        raise ValueError("Non-binary values found in Y")
    
    category_names = Y.columns.tolist()
    return X, Y, category_names

def tokenize(text):
    """
    Tokenize text.
    
    Args:
        text: string. Text to tokenize.
        
    Returns:
        clean_tokens: list. List of clean tokens extracted from the text.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]
    return clean_tokens

def build_model():
    """
    Build machine learning pipeline and perform grid search.
    
    Returns:
        cv: GridSearchCV. Grid search model object.
    """
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize, token_pattern=None)),  # Avoid token_pattern warning
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__min_samples_split': [2, 4]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3, n_jobs=-1)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model performance.
    
    Args:
        model: model object. Trained model.
        X_test: dataframe. Test features.
        Y_test: dataframe. Test labels.
        category_names: list. List of category names.
    """
    Y_pred = model.predict(X_test)
    for i, category in enumerate(category_names):
        print(f'Category: {category}\n')
        print(classification_report(Y_test.iloc[:, i], Y_pred[:, i]))

def save_model(model, model_filepath):
    """
    Save trained model as a pickle file.
    
    Args:
        model: model object. Trained model.
        model_filepath: string. Filepath for output pickle file.
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        print(f'Loading data...\n    DATABASE: {database_filepath}')
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print(f'Saving model...\n    MODEL: {model_filepath}')
        save_model(model, model_filepath)

        print('Trained model saved!')
    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()
