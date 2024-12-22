# Vectorization
import tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

vectorizer = CountVectorizer(tokenizer=tokenize, token_pattern=None)
X_train = ["sample text 1", "sample text 2"]  # Example data
Y_train = [0, 1]  # Example labels
X_vectorized = vectorizer.fit_transform(X_train)

# TF-IDF Transformation
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_vectorized)

# Model Fitting
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf.fit(X_tfidf, Y_train)
