import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def load_and_transform(table_name):
    df = pd.read_table(table_name)

    # Extract some basic features.
    df['word_count'] = df['text_a'].apply(lambda x: len(str(x).split(' ')))
    df['char_count'] = df['text_a'].apply(lambda x: len(x))
    df['hashtag_count'] = df['text_a'].apply(lambda x: x.count('#'))
    df['http_link_count'] = df['text_a'].apply(lambda x: x.count('https://'))
    df['http_link_count'] = df['text_a'].apply(lambda x: x.count('http://'))
    df['number_of_nums'] = df['text_a'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
    df['number_of_non_ascii'] = df['text_a'].apply(lambda x: len(x) - len([i for i in x if ord(i) < 256]))

    # Convert to lower case.
    df['text_a'] = df['text_a'].apply(lambda x: x.lower())

    # Remove urls.
    remove_urls = lambda x: re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x)
    df['text_a'] = df['text_a'].apply(remove_urls)

    # Remove non alpha and not ascii chars.
    remove_non_alpha = lambda x: ''.join([i for i in x if (i.isalpha() and ord(i) < 256) or i == ' '])
    df['text_a'] = df['text_a'].apply(remove_non_alpha)

    # Remove stopwords.
    ENGLISH_STOPWORDS = set(stopwords.words('english'))
    remove_stopwords = lambda x: ' '.join([w for w in x.split() if w not in ENGLISH_STOPWORDS])
    df['text_a'] = df['text_a'].apply(remove_stopwords)

    # Stem words.
    stemmer = SnowballStemmer('english')
    stem_words = lambda x: ' '.join(stemmer.stem(token) for token in word_tokenize(x))
    df['text_a'] = df['text_a'].apply(stem_words)

    # Strip.
    df['text_a'] = df['text_a'].apply(lambda x: (' '.join(x.split())).strip())
    return df


def get_tfidf_features(df, show_plots=False):
    def avg_word_len(x):
        word_lens = [len(i) for i in x.split()]
        return sum(word_lens) / len(word_lens)

    df['avg_word_len'] = df['text_a'].apply(avg_word_len)

    # Bag of words.
    vec = CountVectorizer(max_features=1000, min_df=5, max_df=0.85, ngram_range=(1, 2), analyzer='word')
    X = vec.fit_transform(df['text_a']).toarray()

    if show_plots:
        td_matrix = pd.DataFrame(X, columns=vec.get_feature_names_out())
        td_matrix = td_matrix.T
        td_matrix['total_count'] = td_matrix.sum(axis=1)
        td_matrix = td_matrix.sort_values(by='total_count', ascending=False)[:35]
        td_matrix['total_count'].plot.bar()
        plt.show()

    # TFIDF
    tfidfconverter = TfidfTransformer()
    X = tfidfconverter.fit_transform(X).toarray()
    return X


def get_contextual_features(df):
    return df[[
        'word_count',
        'char_count',
        'hashtag_count',
        'http_link_count',
        'http_link_count',
        'number_of_nums',
        'number_of_non_ascii',
        'avg_word_len',
    ]].to_numpy()


# Load data.
train = load_and_transform('data/train_data.tsv')
x_train_tfidf = get_tfidf_features(train)
x_train_context = get_contextual_features(train)
x_train = np.hstack((x_train_tfidf, x_train_context))
y_train = train['label']

test = load_and_transform('data/test_data.tsv')
x_test_tfidf = get_tfidf_features(test)
x_test_context = get_contextual_features(test)
x_test = np.hstack((x_test_tfidf, x_test_context))
y_test = test['label']


def evaluate_model(model, x_test_against):
    y_pred = model.predict(x_test_against)
    return 'Accuracy: {:.3f}\t\tF1: {:.3f}'.format(accuracy_score(y_test, y_pred), f1_score(y_test, y_pred))


# Works the same regardless of features.
print('========== MAJORITY ==========')
majority_classifier = DummyClassifier()
majority_classifier.fit(x_train_tfidf, y_train)
print(evaluate_model(majority_classifier, x_test_tfidf))

print('========== Knn (tfidf) ==========')
knn_classifier_tfidf = KNeighborsClassifier(n_neighbors=100, metric='minkowski')
knn_classifier_tfidf.fit(x_train_tfidf, y_train)
print(evaluate_model(knn_classifier_tfidf, x_test_tfidf))

print('========== Knn (context) ==========')
knn_classifier_context = KNeighborsClassifier(n_neighbors=3, metric='minkowski')
knn_classifier_context.fit(x_train_context, y_train)
print(evaluate_model(knn_classifier_context, x_test_context))

print('========== SVM linear (tfidf) ==========')
svm_linear_classifier_tfidf = SVC(kernel='linear')
svm_linear_classifier_tfidf.fit(x_train_tfidf, y_train)
print(evaluate_model(svm_linear_classifier_tfidf, x_test_tfidf))

print('========== SVM linear (context) ==========')
svm_linear_classifier_context = SVC(kernel='linear')
svm_linear_classifier_context.fit(x_train_context, y_train)
print(evaluate_model(svm_linear_classifier_context, x_test_context))

print('========== SVM linear (all) ==========')
svm_linear_classifier_all = SVC(kernel='linear')
svm_linear_classifier_all.fit(x_train, y_train)
print(evaluate_model(svm_linear_classifier_all, x_test))

print('========== SVM poly (tfidf) ==========')
svm_poly_classifier_tfidf = SVC(kernel='poly')
svm_poly_classifier_tfidf.fit(x_train_tfidf, y_train)
print(evaluate_model(svm_poly_classifier_tfidf, x_test_tfidf))

print('========== SVM poly (context) ==========')
svm_poly_classifier_context = SVC(kernel='poly')
svm_poly_classifier_context.fit(x_train_context, y_train)
print(evaluate_model(svm_poly_classifier_context, x_test_context))

print('========== SVM poly (all) ==========')
svm_poly_classifier_all = SVC(kernel='poly')
svm_poly_classifier_all.fit(x_train, y_train)
print(evaluate_model(svm_poly_classifier_all, x_test))

print('========== RANDOM FOREST (tfidf) ==========')
rf_classifier_tfidf = RandomForestClassifier(n_estimators=1000, random_state=0)
rf_classifier_tfidf.fit(x_train_tfidf, y_train)
print(evaluate_model(rf_classifier_tfidf, x_test_tfidf))

print('========== RANDOM FOREST (context) ==========')
rf_classifier_context = RandomForestClassifier(n_estimators=30, random_state=0)
rf_classifier_context.fit(x_train_context, y_train)
print(evaluate_model(rf_classifier_context, x_test_context))

print('========== RANDOM FOREST (all) ==========')
rf_classifier_all = RandomForestClassifier(n_estimators=1000, random_state=0)
rf_classifier_all.fit(x_train, y_train)
print(evaluate_model(rf_classifier_all, x_test))

print('========== XGBoost (tfidf) ==========')
xgb_classifier_tfidf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_classifier_tfidf.fit(x_train_tfidf, y_train)
print(evaluate_model(xgb_classifier_tfidf, x_test_tfidf))

print('========== XGBoost (context) ==========')
xgb_classifier_context = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_classifier_context.fit(x_train_context, y_train)
print(evaluate_model(xgb_classifier_context, x_test_context))

print('========== XGBoost (all) ==========')
xgb_classifier_all = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_classifier_all.fit(x_train, y_train)
print(evaluate_model(xgb_classifier_all, x_test))


class CustomVotingClassifier:
    def __init__(self, tfidf_classifiers, context_classifiers, all_classifiers):
        self.tfidf_classifiers = tfidf_classifiers
        self.context_classifiers = context_classifiers
        self.all_classifiers = all_classifiers
        self.len = len(tfidf_classifiers) + len(context_classifiers) + len(all_classifiers)

    def predict(self, X):
        res = np.zeros(X.shape[0])
        for c in self.tfidf_classifiers:
            y_pred = c.predict(x_test_tfidf)
            res = np.add(res, y_pred)

        for c in self.context_classifiers:
            y_pred = c.predict(x_test_context)
            res = np.add(res, y_pred)

        for c in self.all_classifiers:
            y_pred = c.predict(x_test)
            res = np.add(res, y_pred)

        def vote(x):
            a = x / self.len
            return 0 if a < 0.5 else 1

        vf = np.vectorize(vote)
        return vf(res)


print('========== Custom voting ==========')
custom_voting_classifier = CustomVotingClassifier(
    [knn_classifier_tfidf, svm_linear_classifier_tfidf, svm_poly_classifier_tfidf, rf_classifier_tfidf,
     xgb_classifier_tfidf],
    [knn_classifier_context, svm_linear_classifier_context, svm_poly_classifier_context, rf_classifier_context,
     xgb_classifier_context],
    [svm_linear_classifier_all, svm_poly_classifier_all, rf_classifier_all, xgb_classifier_all]
)
print(evaluate_model(custom_voting_classifier, x_test_context))
