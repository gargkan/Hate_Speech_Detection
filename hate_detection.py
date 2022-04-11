import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer# To convert text into numerics

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC # Machine Learning Algorithm
import re
import nltk
from nltk.corpus import stopwords
import string
from sklearn.metrics import accuracy_score
from sklearn import metrics

nltk.download('stopwords')

stemmer = nltk.SnowballStemmer("english")

stopword=set(stopwords.words('english'))
data = pd.read_csv("labeled_data.csv")

data["labels"] = data["class"].map({0: "Hate Speech", 
                                    1: "Offensive Language", 
                                    2: "Neither"})
data = data[["tweet", "labels"]]

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
data["tweet"] = data["tweet"].apply(clean)

x = np.array(data["tweet"])
y = np.array(data["labels"])

cv = TfidfVectorizer()
X = cv.fit_transform(x) # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = SVC(kernel='rbf', random_state = 1)
clf.fit(X_train,y_train)

ypredict = clf.predict( X_test)
print(ypredict)

print(metrics.classification_report(y_test,ypredict))

accuracy_score(y_test, ypredict)

import matplotlib.pyplot as plt
from collections import Counter
hate_frequency = Counter(y)
plt.bar(hate_frequency.keys(), hate_frequency.values())
plt.show()
