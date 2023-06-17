from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


def svm(games_df, x_train, x_test, y_train, y_test):
    model = SVC(kernel='rbf')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, y_test, y_pred
