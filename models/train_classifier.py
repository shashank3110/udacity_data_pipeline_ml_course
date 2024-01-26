# import libraries
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])
import sys
import time
import pandas as pd
import re
from sklearn.model_selection import GridSearchCV,StratifiedKFold,train_test_split
from sklearn.metrics import accuracy_score, f1_score, fbeta_score, confusion_matrix, classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline
import sqlite3
from sqlalchemy import create_engine
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

def load_data(database_filepath):
    # load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('disaster_data_cleaned', engine)
    
    X = df[df.columns[:4]]
    Y = df[df.columns[4:]]
    
    return X['message'], Y, Y.columns
    


def tokenize(text):
    
    # make it case agnostic
    text = text.lower()
    # replace punctuations with spaces
    text = re.sub(r'[^[a-zA-Z0-9]]', " ", text)
    
    # get word tokens
    tokens = word_tokenize(text)
    
    # remove stop words
    words = [word for word in tokens if word not in stopwords.words('english')]
    
    return words


def build_model():
    
    # obtain train test split
    #X_train, X_test, y_train, y_test = train_test_split(X['message'],Y)
    
    # build feature + model pipeline
    pipeline = Pipeline([
    
   ('features',TfidfVectorizer(tokenizer=tokenize)),
    
    ('clf', MultiOutputClassifier(estimator=RandomForestClassifier()))])
    
    
    # setup gridsearch CV for hyperparam tuning
    parameters = {'clf__estimator__max_depth':[100, 150],
             'clf__estimator__n_estimators':[10, 50]
             
             }
    cv_model = GridSearchCV(estimator=pipeline, param_grid=parameters)

    # train feature + model pipleine with Grid search cross validation
    #model = cv.fit(X_train,y_train)
    return cv_model
    

def evaluate_model(model, X_test, Y_test, category_names):
    
    # get predictions from the trained model
    Y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred, columns=category_names)
    for col in category_names:
        print(col, classification_report(Y_test[col], Y_pred[col]))



def save_model(model, model_filepath):
    joblib.dump(model,f'{model_filepath}')


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training models...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()