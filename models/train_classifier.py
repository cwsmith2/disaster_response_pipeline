import sys
import pandas as pd
import pickle
from sklearn import pipeline
from sqlalchemy import create_engine
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from lightgbm import LGBMClassifier
import nltk
nltk.download(['punkt', 'wordnet','punkt_tab'])

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ml_pipeline.log"),
        logging.StreamHandler()
    ]
)

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
    logging.info(f"Loading data from {database_filepath}")
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('DisasterMessages', engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    
    # Find problematic columns
    non_binary_columns = Y.columns[~Y.applymap(lambda x: x in [0, 1]).all()]
        
    # Raise error if non-binary values exist
    if not non_binary_columns.empty:
        logging.warning(f"Non-binary values found in columns: {non_binary_columns.tolist()}")
        Y = Y.applymap(lambda x: 1 if x > 0 else 0)
    
    # Filter out categories with a single unique value
    single_class_columns = [col for col in Y.columns if Y[col].nunique() == 1]
    if single_class_columns:
        print(f"Removing single-class columns: {single_class_columns}")
        Y = Y.drop(columns=single_class_columns)

    category_names = Y.columns.tolist()
    logging.info(f"Loaded {X.shape[0]} records and {Y.shape[1]} categories.")
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
    logging.info("Starting to build the machine learning pipeline.")
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, token_pattern=None)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(LGBMClassifier(force_row_wise=True, verbose=-1)))
    ])
    logging.info("Pipeline structure defined.")

    parameters = {
        'clf__estimator__boosting_type': ['gbdt'],
        'clf__estimator__num_leaves': [31],
        'clf__estimator__n_estimators': [50],
        'clf__estimator__learning_rate': [0.1],
        'clf__estimator__min_data_in_leaf': [1, 5],  # Allow smaller leaf sizes
        'clf__estimator__scale_pos_weight': [1, 2]  # Adjust for imbalanced classes
    }        
    logging.info(f"Hyperparameters for RandomizedSearchCV: {parameters}")

    cv = RandomizedSearchCV(
        pipeline,
        param_distributions=parameters,
        n_iter=10,
        cv=3,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    logging.info("RandomizedSearchCV initialized.")
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
    logging.info("Starting model evaluation.")
    try:
        Y_pred = model.predict(X_test)
        for i, category in enumerate(category_names):
            report = classification_report(Y_test.iloc[:, i], Y_pred[:, i], zero_division=0)
            logging.info(f"Category: {category}\n{report}")
        logging.info("Model evaluation completed successfully.")
    except Exception as e:
        logging.error(f"Error during evaluation: {e}")

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
    try:
        if len(sys.argv) == 3:
            database_filepath, model_filepath = sys.argv[1:]

            logging.info(f"Loading data from {database_filepath}")
            X, Y, category_names = load_data(database_filepath)

            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
            logging.info(f"Data split into training and test sets: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples.")

            logging.info("Building the model.")
            model = build_model()

            logging.info("Training the model.")
            model.fit(X_train, Y_train)
            logging.info("Model training completed.")

            logging.info("Evaluating the model.")
            evaluate_model(model, X_test, Y_test, category_names)

            logging.info(f"Saving the model to {model_filepath}.")
            save_model(model, model_filepath)
            logging.info("Model saved successfully.")
        else:
            logging.error("Invalid number of arguments provided.")
            print('Usage: python train_classifier.py database_filepath model_filepath')
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
