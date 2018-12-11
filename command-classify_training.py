import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import numpy as np


# nạp dữ liệu
df = pandas.read_csv("fruitg.csv")
Xpre = df['SENTENCE']
Y = df['COMMAND']

# tách dữ liệu thành biểu diễn TF-IDF
tfidf_converter = TfidfVectorizer(max_features=150)
X = tfidf_converter.fit_transform(Xpre).toarray()

# tách bộ dữ liệu train và test 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# khởi tạo mô hình Random Forest
classifier = RandomForestClassifier(n_estimators=100, random_state=0)
# khớp dữ liệu
# classifier.fit(X_train, Y_train)
classifier.fit(X, Y)

# đánh giá mô hình
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# print (confusion_matrix(Y_test, Y_pred))
# print (classification_report(Y_test, Y_pred))
# print (accuracy_score(Y_test, Y_pred))

# lưu lại mô hình vào tệp command-classifier
with open('command-classifier', 'wb') as picklefile:
    pickle.dump(classifier, picklefile)