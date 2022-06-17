import pickle
import sklearn
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

# Load and prepare data
with open(file="scikit-learn-data.pickle", mode="rb") as file:
    sklearn_data = pickle.load(file)

x_train = sklearn_data['x_train']
y_train = sklearn_data['y_train']
x_test = sklearn_data['x_test']
y_test = sklearn_data['y_test']

features = 2**5
vectorizer = HashingVectorizer(n_features = features)
X_train = vectorizer.fit_transform(x_train)
X_test = vectorizer.fit_transform(x_test)

# Naive Bayes
gnb = GaussianNB()
X_train = X_train.toarray()
X_test = X_test.toarray()
gnb.fit(X_train, y_train)

print('Accuracy on test set using Naive Bayes:', sklearn.metrics.accuracy_score(y_test, gnb.predict(X_test)))
print('Accuracy on training set using Naive Bayes:', sklearn.metrics.accuracy_score(y_train, gnb.predict(X_train)))

# Decision tree
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

print('Accuracy on test set using default Decision tree:', sklearn.metrics.accuracy_score(y_test, clf.predict(X_test)))
print('Accuracy on training set using default Decision tree:', sklearn.metrics.accuracy_score(y_train, clf.predict(X_train)))


# Randomized search to find the better parameters
# param_dist = {"max_depth": [None, 5, 10, 20],
#               "max_features": [None, 'log2', 'sqrt'],
#               "min_samples_leaf": [1, 2, 3, 4, 5, 6, 8, 9],
#               "criterion": ["gini", "entropy"]}
# tree = DecisionTreeClassifier()
# tree_cv = RandomizedSearchCV(tree, param_dist, n_iter = 15, cv=5, random_state = 1, verbose = 1, scoring='accuracy')

# tree_cv.fit(X_train,y_train)

# print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
# print("Best score is {}".format(tree_cv.best_score_))

# best_params_ = tree_cv.best_params_

# Better parameters for decision tree (which was found using the above code)
best_params_ = {'min_samples_leaf': 3, 'max_features': None, 'max_depth': 5, 'criterion': 'gini'}

tree_tuned = DecisionTreeClassifier(**best_params_)
tree_tuned.fit(X_train, y_train)

print('Accuracy on training set using tuned Decision tree:', sklearn.metrics.accuracy_score(y_train, tree_tuned.predict(X_train)))
print('Accuracy on test set using tuned Decision tree:', sklearn.metrics.accuracy_score(y_test, tree_tuned.predict(X_test)))

# Checking what happens if we increase max_depth to 20
best_params_ = {'min_samples_leaf': 3, 'max_features': None, 'max_depth': 20, 'criterion': 'gini'}

tree_tuned = DecisionTreeClassifier(**best_params_)
tree_tuned.fit(X_train, y_train)

print('Accuracy on training set using tuned Decision tree with max depth 20:', sklearn.metrics.accuracy_score(y_train, tree_tuned.predict(X_train)))
print('Accuracy on test set using tuned Decision tree with max depth 20:', sklearn.metrics.accuracy_score(y_test, tree_tuned.predict(X_test)))

# Plot the decision tree
# plt.figure(figsize = (20,10))
# tree.plot_tree(tree_tuned,
#               filled=True, 
#               rounded=True, 
#               fontsize=5)
# plt.show()
