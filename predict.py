import gensim, os
from joblib import load

# load models
count_vect = load('./models/CountVectorizer.joblib')
tfidf_transformer = load('./models/TfidfTransformer.joblib')
clf = load('./models/bestModel.joblib')

def clean_doc(text, vocab):
    """ turn a text doc into clean tokens

    Args:
        text (Pandas Series): Uncleaned text
        vocab ([type]): vacab from tranining data

    Returns:
        ndarray: cleaned text
    """

    # to lower case and get rid of punctuations using regex
    # further clean the text data, keeping own words in the training vocab

    if type(text) == str: # single string
        text = text.lower()
        text = text.replace('[^\w\s]', '')
        return ' '.join([w for w in text.split() if w in vocab])
    else: # list of strings
        text = pd.Series(text)
        text = text.str.lower()
        text = text.str.replace('[^\w\s]', '')
        cleaned = []
        for x in text.values:
            cleaned.append(' '.join([w for w in x.split() if w in vocab]))
    
    return np.array(cleaned)

def load_doc(filename):
    """ load doc into memory

    Args:
        filename (String): file path

    Returns:
        list: list of Strings
    """
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

def predict(text, count_vect, tfidf_transformer, clf):
    """Predict sentiment

    Args:
        text (String): Input text
        count_vect (sklearn transformer): turn into tokens
        tfidf_transformer (sklearn transformer): apply tfdif
        clf (sklearn model): linear svm model
    """
    X_test_counts = count_vect.transform(text)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    y_pred = clf.predict(X_test_tfidf)
    return y_pred

def sentiment(text):
    if len(text):
        # load vocab dictionary
        vocab_filename = './data/cleaned/vocab.txt'
        vocab = load_doc(vocab_filename)
        vocab = set(vocab.split())

        # clean test data
        cleaned = clean_doc(text, vocab)
        print('predicte here')

        return predict([cleaned], count_vect, tfidf_transformer, clf)[0]
    return None

