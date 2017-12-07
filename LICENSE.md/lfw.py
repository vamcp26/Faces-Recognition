import numpy as np
import pandas as pd
from sklearn.datasets import fetch_lfw_people
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

X_people=fetch_lfw_people(min_faces_per_person=70,resize=0.25)

x=X_people.data
y=X_people.target

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.40, random_state=25)

pca=PCA(n_components=250).fit(x_train)
x_train_pca=pca.transform(x_train)
x_test_pca=pca.transform(x_test)


logistic = LogisticRegression()
logistic.fit(x_train_pca,y_train)
logistic.predict(x_test_pca)
print(cross_val_score(logistic,x_test_pca,y_test))
