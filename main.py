import matplotlib.pyplot as plt
import pandas as pd
from nltk import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB


def remove_stopwords(text):
    stop = set(stopwords.words('english'))
    text = [word for word in text.split() if word not in stop]
    text = ' '.join(x for x in text)
    return text
stemmer = SnowballStemmer("english")
def stemm_words(text):
    text = [stemmer.stem(word) for word in text.split()]
    text = ' '.join(x for x in text)
    return text

# Feature extraction
# load the dataset
data = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
# extract only the text from data
df_train = pd.DataFrame(data.data[:100], columns=["text"])
# remove stopwords and stemming
df_train['text'] = df_train['text'].apply(remove_stopwords)
df_train['text'] = df_train['text'].apply(stemm_words)
# remove punctuaction
df_train["text"] = df_train['text'].str.replace('[^\w\s]', '')
# remove numbers
df_train['text'] = df_train['text'].str.replace('\d+', '')
# remove whitespaces
df_train["text"] = df_train["text"].str.strip()
# count some features
df_train["num_char"] = df_train["text"].str.strip().str.len()
df_train["num_words"] = df_train["text"].str.split().str.len()
df_train["num_vocab"] = df_train["text"].str.lower().str.split().apply(set).str.len()
df_train['lexical_div'] = df_train['num_words'] / df_train['num_vocab']
df_train['ave_word_length'] = df_train['num_char'] / df_train['num_words']
# select the target
df_train["target_class"] = data.target[:100]

# Plot bar chart to check number of elements in a class
cols = df_train.columns.to_list()
target = df_train.target_class.value_counts().sort_index()
target.plot.bar()
plt.show()
# Tokenize the text
vectorizer = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 1))
vectorizer.fit(df_train['text'])
X = vectorizer.transform(df_train['text'])
bagofwords = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
#print(df_train.head())
#print(bagofwords.head())
df_train = pd.concat([df_train, bagofwords], axis=1)
X_train = df_train.loc[:, df_train.columns != 'target_class']
Y_train = df_train["target_class"]
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
print(X_train)


data = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"))
# extract only the text from data
df_test = pd.DataFrame(data.data[:100], columns=["text"])
# remove stopwords and stemming
df_test['text'] = df_test['text'].apply(remove_stopwords)
df_test['text'] = df_test['text'].apply(stemm_words)
# remove punctuaction
df_test["text"] = df_test['text'].str.replace('[^\w\s]', '')
# remove numbers
df_test['text'] = df_test['text'].str.replace('\d+', '')
# remove whitespaces
df_test["text"] = df_test["text"].str.strip()
# count some features
df_test["num_char"] = df_test["text"].str.strip().str.len()
df_test["num_words"] = df_test["text"].str.split().str.len()
df_test["num_vocab"] = df_test["text"].str.lower().str.split().apply(set).str.len()
df_test['lexical_div'] = df_test['num_words'] / df_test['num_vocab']
df_test['ave_word_length'] = df_test['num_char'] / df_test['num_words']
# select the target
df_test["target_class"] = data.target[:100]

# Plot bar chart to check number of elements in a class
cols = df_test.columns.to_list()
target = df_test.target_class.value_counts().sort_index()
target.plot.bar()
plt.show()
# Tokenize the text
vectorizer = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 1))
vectorizer.fit(df_test['text'])
X = vectorizer.transform(df_test['text'])
bagofwords = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
print(df_test.head())
print(bagofwords.head())
df_test = pd.concat([df_test, bagofwords], axis=1)
X_test = df_test.loc[:, df_test.columns != 'target_class']
y_test = df_test["target_class"].to_numpy()

X_test = vectorizer.fit_transform(X_test)





params = {"alpha": (1.0, 0.1, 1e-2, 1e-3)}
clf = GridSearchCV(MultinomialNB(), params)
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=data.target_names))
print(f"F1 {f1_score(y_test, y_pred, average='weighted'):.3f}")

