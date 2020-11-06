"""
Train Classifier Script
To run the script, type or copy/pase below line:
> python train_classifier.py ../data/DisasterResponse.db model.p
Arguments:
    1) Input File 1: DisasterResponse_Processed.db - SQLite database
    2) Input File 2: ML Model                      - pickle file
"""

import sys
import nltk
nltk.download(['punkt', 'wordnet'])
import nltk
nltk.download('averaged_perceptron_tagger')

import re
import pickle
import pandas as pd
import numpy  as np
from nltk.tokenize import word_tokenize, WhitespaceTokenizer
from nltk.stem     import WordNetLemmatizer
from sqlalchemy    import create_engine

from sklearn.pipeline                import Pipeline, FeatureUnion
from sklearn.multioutput             import MultiOutputClassifier
from sklearn.ensemble                import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.model_selection         import train_test_split, GridSearchCV
from sklearn.metrics                 import make_scorer, accuracy_score, f1_score, fbeta_score, classification_report
from scipy.stats.mstats              import gmean
from sklearn.base                    import BaseEstimator, TransformerMixin

def load_data(database_filepath):
    """
    Load Data Function does following:
    1. reads input SQLite database generated by ETL
    2. splits dataframe into Feature & Label
    
    Arguments:
        database_filepath - SQLite database
    Output:
        X              - Feature Dataframe
        Y              - Label   Dataframe
        category_names - used for data visualization (app)
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df     = pd.read_sql_table('message_categories', engine)
    X = df['message']
    y = df.iloc[:,4:]
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    """
    Tokenize function does following:
    1. tokenization function to process text data
       including lemmatize, normalize case, and remove leading/trailing white space
    
    Arguments:
        text         - list of text messages (english)
    Output:
        clean_tokens - tokenized text, clean and ready to feed ML modeling
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)
    
    return clean_tokens    


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    StartingVerb Extractor class
    
    This class extract the starting verb of a sentence,
    creating a new feature for the ML classifier
    """
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def multioutput_fscore(y_true,y_pred,beta=1):
    """
    MultiOutput Fscore
    
    This is a performance metric reference from Github User: matteobonanomi and xiaoye318024
    It is a sort of geometric mean of the fbeta_score, computed on each label.
    
    It is compatible with multi-label and multi-class problems.
    It features some peculiarities (geometric mean, 100% removal...) to exclude
    trivial solutions and deliberatly under-estimate a stangd fbeta_score average.
    The aim is avoiding issues when dealing with multi-class/multi-label imbalanced cases.
    
    It can be used as scorer for GridSearchCV:
        scorer = make_scorer(multioutput_fscore,beta=1)
        
    Arguments:
        y_true -> labels
        y_pred -> predictions
        beta -> beta value of fscore metric
    
    Output:
        f1score -> customized fscore
    """
    score_list = []
    if isinstance(y_pred, pd.DataFrame) == True:
        y_pred = y_pred.values
    if isinstance(y_true, pd.DataFrame) == True:
        y_true = y_true.values
    for column in range(0,y_true.shape[1]):
        score = fbeta_score(y_true[:,column],y_pred[:,column],beta,average='weighted')
        score_list.append(score)
    f1score_numpy = np.asarray(score_list)
    f1score_numpy = f1score_numpy[f1score_numpy<1]
    f1score = gmean(f1score_numpy)
    return  f1score


def build_model():
    """
    Build Model function
    
    This function returns a ML Pipeline that process text messages
    according to NLP best-practice and apply a classifier.
    """
    pipeline = Pipeline([('features', FeatureUnion([('text_pipeline', Pipeline([('vect',  CountVectorizer(tokenizer=tokenize)),
                                                                                ('tfidf', TfidfTransformer()),
                                                                               ])),
                                                    ('starting_verb', StartingVerbExtractor())
                                                   ])
                         ),
                         ('clf', MultiOutputClassifier(AdaBoostClassifier()))
                       ])

    #Use grid search to find better parameters
    parameters = {
               'clf__estimator__n_estimators': [10]
#             'features__text_pipeline__vect__ngram_range':  ((1, 1), (1, 2)),
#             'features__text_pipeline__vect__max_df':       (0.5, 0.75, 1.0),
#             'features__text_pipeline__vect__max_features': (None, 5000, 10000),
#             'features__text_pipeline__tfidf__use_idf':     (True, False),
    }

    scorer = make_scorer(multioutput_fscore,greater_is_better = True)
    cv = GridSearchCV(pipeline, param_grid = parameters, scoring = scorer,verbose = 2, n_jobs = -1)
    return cv


def evaluate_model(model, X_test, y_test, category_names):
    """
    Evaluate Model function
    
    This function use the provided ML pipeline to predict on a test set
    and report the f1 score, precision and recall for each output category of the dataset
    
    Arguments:
        model          - ML Pipeline
        X_test         - test features
        y_test         - test labels
        category_names - label names (multi-output)
    """
    y_pred = model.predict(X_test)

    for i in range(len(category_names)):
        print('------------------------------------------------------\n')
        print("Category:", category_names[i],"\n", classification_report(y_test.iloc[:, i], y_pred[:, i]))

    multioutput_f1   = multioutput_fscore(y_test, y_pred, beta = 1)
    overall_accuracy = (y_pred == y_test).mean().mean()
    print('Average overall accuracy:     {0:.2f}%\n'.format(overall_accuracy*100))
    print('F1 score (custom definition): {0:.2f}%\n'.format(multioutput_f1*100))


def save_model(model, model_filepath):
    """
    Save Model function
    
    This function saves trained model as Pickle file, to be loaded later.
    
    Arguments:
        model          - GridSearchCV or Scikit Pipeline object
        model_filepath - output pickle file path & name    
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """
    Train Classifier Main function
    
    This function applies the Machine Learning Pipeline:
        1) Extract data from SQLite db
        2) Train ML model on training set
        3) Estimate model performance on test set
        4) Save trained model as Pickle
    
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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
