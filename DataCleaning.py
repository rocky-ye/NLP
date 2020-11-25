import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from collections import Counter
from sklearn.model_selection import train_test_split


def clean(text):
    """ Normalize and clean text and generate a list of vocab

    Args:
        text (Pansas Series): Review discriptions

    Returns:
        list: Cleaned Review discriptions
        list: Clean list of vocabularies
        
    """
    
    tokens = []
    # to lower case
    text = text.str.lower()
    # get rid of punctuations using regex
    text = text.str.replace('[^\w\s]', '')

    # get rid of numbers, stop words, one-letter words
    for x in text.values.flatten():
        tokens.extend(x.split())
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words and len(word) > 1]

    # keep tokens with freq > 2
    vocab = Counter()
    vocab.update(tokens)
    min_occurane = 2
    tokens = set([k for k,c in vocab.items() if c >= min_occurane])

    # save tokens
    save_list(tokens, 'data/cleaned/vocab.txt')

    # further clean the text data, keeping own words in the vocab
    doc = []
    for x in text.values:
        doc.append(' '.join([w for w in x.split() if w in tokens]))
    
    return doc, tokens



def save_list(lines, filename):
    """ save vocab to filepath

    Args:
        lines (list): Clean list of vocabularies
        filename (String): file path
    """

    # convert lines to a single blob of text
    data = '\n'.join(lines)
    # open file
    file = open(filename, 'w')
    # write text
    file.write(data)
    # close file
    file.close()
 


# load data
text = pd.read_csv('data/raw/Reviews.csv')

# create binary target variable
text['Target'] = np.where(text.Score < 4, 0, 1)

# split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(text.Text, text.Target, test_size=0.25, random_state=0)

# process training data
X_train, tokens = clean(X_train)

# save tokens and cleaned training data 
save_list(tokens, 'data/cleaned/vocab.txt')
pd.DataFrame(np.c_[X_train, y_train]).to_csv('data/cleaned/Training.csv', index=False)
