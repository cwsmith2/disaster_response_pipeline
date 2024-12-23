
# Disaster Response Pipeline Project

## Table of Contents
1. [Project Overview](#project-overview)
2. [Getting Started](#getting-started)
   - [Dependencies](#dependencies)
   - [Installing and Setting Up](#installing-and-setting-up)
3. [Project Structure](#project-structure)
4. [Instructions](#instructions)
   - [Running ETL Pipeline](#running-etl-pipeline)
   - [Running ML Pipeline](#running-ml-pipeline)
   - [Running the Web App](#running-the-web-app)
5. [Features of the Web App](#features-of-the-web-app)
6. [Additional Resources](#additional-resources)
7. [Authors](#authors)
8. [License](#license)
9. [Acknowledgements](#acknowledgements)
10. [Screenshots](#screenshots)

---

## Project Overview

The **Disaster Response Pipeline Project** is part of the Udacity Data Science Nanodegree program in collaboration with Figure Eight. The goal of this project is to build a machine learning pipeline capable of categorizing real-life disaster messages. The categorized messages can then be directed to the appropriate disaster relief agency.

### Key Components:
- **ETL Pipeline**: Extracts data from a dataset of real-life disaster messages, cleans and processes it, and stores it in an SQLite database.
- **Machine Learning Pipeline**: Builds and trains a classifier using NLP techniques to predict the categories of input messages.
- **Interactive Web App**: Allows users to input a message and receive real-time classifications, alongside visualizations of the training dataset.

---

## Getting Started

### Dependencies
This project requires the following Python libraries and tools:
- **Python 3.5+**
- **Data Manipulation**: pandas, NumPy
- **Machine Learning**: scikit-learn
- **Natural Language Processing**: nltk
- **Database Management**: sqlalchemy
- **Web Framework**: Flask
- **Data Visualization**: Plotly
- **Model Saving**: Pickle

### Installing and Setting Up
1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/cwsmith2/disaster-response-pipeline.git
   cd disaster-response-pipeline
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Project Structure

```
disaster-response-pipeline/
├── app/
│   ├── run.py               # Runs the Flask web app
│   ├── templates/           # HTML templates for the web app
├── data/
│   ├── disaster_messages.csv   # Raw dataset containing messages
│   ├── disaster_categories.csv # Categories for messages
│   ├── process_data.py         # ETL pipeline script
├── models/
│   ├── train_classifier.py     # Machine learning pipeline script
│   ├── classifier.pkl          # Trained model (generated after running ML pipeline)
└── README.md
```

---

## Instructions

### Running ETL Pipeline
To clean the dataset and store it in a SQLite database:
```bash
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```

### Running ML Pipeline
To train the classifier and save it as a `.pkl` file:
```bash
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```

### Running the Web App
To start the Flask web application:
```bash
python app/run.py
```
Then open your browser and navigate to: [http://0.0.0.0:3001/](http://0.0.0.0:3001/)

---

## Features of the Web App

1. **Real-Time Message Classification**:
   - Input a message into the web app.
   - Predicts the disaster categories (e.g., medical help, shelter, water, etc.).
   
2. **Training Data Insights**:
   - Visualizations of message distribution by genre (social, news, direct).
   - Most frequently occurring categories.

3. **Error Handling**:
   - Interactive examples guide users on valid message inputs.


## Authors
- **CW Smith**
- GitHub: [Your Profile](https://github.com/cwsmith2)

---

## License
This project is licensed under the **MIT License**.

---

## Acknowledgements
- **Udacity**: For the Data Science Nanodegree curriculum and guidance.
- **Figure Eight**: For providing the dataset.
- **Open Source Libraries**: For the tools and libraries used in this project.

---