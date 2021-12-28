import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt


def load_and_transform(table_name):
    df = pd.read_table(table_name)

    # Convert to lower case.
    df['text_a'] = df['text_a'].apply(lambda x: x.lower())

    # Remove urls.
    remove_urls = lambda x: re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x)
    df['text_a'] = df['text_a'].apply(remove_urls)

    # Remove non alpha and not ascii chars.
    remove_non_alpha = lambda x: ''.join([i for i in x if (i.isalpha() and ord(i) < 256) or i == ' '])
    df['text_a'] = df['text_a'].apply(remove_non_alpha)
    print(df)

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


df = load_and_transform('data/train_data.tsv')
vec = CountVectorizer()
X = vec.fit_transform(df['text_a'])
td_matrix = pd.DataFrame(X.toarray(), columns=vec.get_feature_names_out())

td_matrix = td_matrix.T
td_matrix['total_count'] = td_matrix.sum(axis=1)
td_matrix = td_matrix.sort_values(by='total_count', ascending=False)[:35]
print(td_matrix)

td_matrix['total_count'].plot.bar()
plt.show()
