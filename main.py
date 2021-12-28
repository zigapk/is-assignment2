import pandas as pd
import nltk
# nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from nltk.tokenize import word_tokenize

df = pd.read_table('data/train_data.tsv')

# Convert to lower case.
df['text_a'] = df['text_a'].apply(lambda x: x.lower())

# Remove urls.
remove_urls = lambda x: re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', x)
df['text_a'] = df['text_a'].apply(remove_urls)

# Remove punctuations and numbers.
PUNCTUATIONS_AND_NUMBERS = '!()-[]{};:\'"\\,<>./?@#$%^&*_~0123456789'
remove_punctuations = lambda x: ''.join([i for i in x if i not in PUNCTUATIONS_AND_NUMBERS])
df['text_a'] = df['text_a'].apply(remove_punctuations)

# Remove stopwords.
ENGLISH_STOPWORDS = set(stopwords.words('english'))
remove_stopwords = lambda x: ' '.join([w for w in x.split() if w not in ENGLISH_STOPWORDS])
df['text_a'] = df['text_a'].apply(remove_stopwords)


print(df)

# Stem words.
stemmer = SnowballStemmer('english')
stem_words = lambda x: ' '.join(stemmer.stem(token) for token in word_tokenize(x))
df['text_a'] = df['text_a'].apply(stem_words)

# Strip.
df['text_a'] = df['text_a'].apply(lambda x: (' '.join(x.split())).strip())
print(df)
