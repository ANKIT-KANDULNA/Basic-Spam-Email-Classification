import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import svm
from sklearn.model_selection import GridSearchCV

# Loading Dataset
dataframe=pd.read_csv("spam_ham_dataset.csv")
print(dataframe.head())
print(dataframe.describe())

# Split into Training and Testing Data
# 80% as training and remaining as testing
x=dataframe["text"]
y=dataframe["label"]
x_train, y_train=x[0:4110],y[0:4110]
x_test, y_test=x[4110:],y[4110:]

# extract features
cv=CountVectorizer()
features=cv.fit_transform(x_train)

# build a model
tuned_parameters={"kernel":["linear","rbf"],"gamma":[1e-3,1e-4],"C":[1,10,100,1000]}
model=GridSearchCV(svm.SVC(),tuned_parameters)
model.fit(features,y_train)
print(model.best_params_)

# test accuracy
feature_test=cv.transform(x_test)
print("Model score is: ",model.score(feature_test,y_test))
