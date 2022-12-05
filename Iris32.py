from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X = load_iris().data
y = load_iris().target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = DecisionTreeClassifier()
fitted = clf.fit(X_train, y_train)

y_pred = fitted.predict(X_test)

score = accuracy_score(y_test, y_pred)
print('Accuracy: ', score)