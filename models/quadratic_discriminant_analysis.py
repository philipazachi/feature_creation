from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score


def quadratic_discriminant_analysis(games_df, x_train, x_test, y_train, y_test):
    model = QuadraticDiscriminantAnalysis()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, y_test, y_pred
