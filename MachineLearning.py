import pandas as pd
import numpy as np

import argparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from joblib import dump


def main():
    # load training data
    df = pd.read_csv('data/cleaned/Training.csv')
    df = df.dropna()
    X_train, y_train = df['0'], df['1']
    # load test data
    df = pd.read_csv('data/cleaned/Test.csv')
    df = df.dropna()

    X_test, y_test = df['0'], df['1']
    # train test split
    # X_train, X_test, y_train, y_test = train_test_split(df.Text, df.Target, test_size=0.25, random_state=0)
    
    # Logistic Regression with 3 different parmas
    # fit_and_test(LogisticRegression(random_state=0, penalty='l2'), 1, True, X_train, y_train, X_test, y_test)
    # fit_and_test(LogisticRegression(random_state=0, penalty='l2'), 2, True, X_train, y_train, X_test, y_test)
    # fit_and_test(LogisticRegression(random_state=0, penalty='l1'), 1, True, X_train, y_train, X_test, y_test)
    # fit_and_test(LogisticRegression(random_state=0, penalty='l1'), 1, False, X_train, y_train, X_test, y_test)
    
    # Linear SVM with 3 different parmas
    # fit_and_test(LinearSVC(random_state=0, loss='squared_hinge'), 1, True, X_train, y_train, X_test, y_test)
    fit_and_test(LinearSVC(random_state=0, loss='squared_hinge'), 2, True, X_train, y_train, X_test, y_test)
    # fit_and_test(LinearSVC(random_state=0, loss='hinge'), 1, True, X_train, y_train, X_test, y_test)
    # fit_and_test(LinearSVC(random_state=0, loss='squared_hinge'), 1, False, X_train, y_train, X_test, y_test)


def fit_and_test(model, n_gram, use_idf, X_train, y_train, X_test, y_test):
    count_vect = CountVectorizer(ngram_range=(1, n_gram))
    tfidf_transformer = TfidfTransformer(use_idf=use_idf)
    
    X_train_counts = count_vect.fit_transform(X_train)
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    
    clf = model.fit(X_train_tfidf, y_train)

    X_test_counts = count_vect.transform(X_test)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    y_pred = clf.predict(X_test_tfidf)
    print('Accuracy is: %f' % accuracy_score(y_test, y_pred))
    print('Recall is: %f' % recall_score(y_test, y_pred))
    print('precision is: %f' % precision_score(y_test, y_pred))

    # persist model
    dump(clf, 'models/bestModel.joblib')
    dump(count_vect, 'models/CountVectorizer.joblib')
    dump(tfidf_transformer, 'models/TfidfTransformer.joblib')


if __name__ == '__main__':
    main()