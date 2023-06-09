from models.svm import svm
from models.quadratic_discriminant_analysis import quadratic_discriminant_analysis
from models.adaboost import adaboost, adaboost_with_finetune
from models.neural_networks import neural_networks
from models.decision_tree import train_decision_tree
from models.gradient_boosting import train_gradient_boosting, train_gradient_boosting_with_finetune
from models.random_forests import train_random_forest, train_random_forest_with_finetune
from models.xgboost import train_xgboost, train_xgboost_with_finetune


def train_all_models(column_name, df, features_names):
    results = {}
    distinct_values = df[column_name].unique()
    accuracy_score = 0
    for category in distinct_values:
        new_df = df[df[column_name] == category]
        print("category", category, "size:", len(new_df))
        x_train, x_test, y_train, y_test = get_train_test(new_df, features_names)
        results[category] = train_models_per_type(new_df, x_train, x_test, y_train, y_test)
        accuracy_score += (results[category][1] * len(new_df) / len(df))
    print("different models:\n", results)
    print("different models accuracy:", accuracy_score)
    x_train, x_test, y_train, y_test = get_train_test(df, features_names)
    result = train_models_per_type(df, x_train, x_test, y_train, y_test)
    print("one model for all:", result)
    return results


def train_models_per_type(games_df, x_train, x_test, y_train, y_test):
    models_options = {
        "decision_tree": train_decision_tree,
        "random_forests": train_random_forest_with_finetune,
        # "gradient_boosting": train_gradient_boosting_with_finetune,
        "adaboost": adaboost_with_finetune,
        "xgboost": train_xgboost_with_finetune,
        # "svm": svm,
        # "QDA": quadratic_discriminant_analysis,
        'neural_networks': neural_networks
    }
    arguments = (games_df, x_train, x_test, y_train, y_test)
    picked_accuracy = 0
    picked_model_name = ""
    for model_name in models_options:
        model_accuracy, model_y_test, model_y_pred = models_options[model_name](*arguments)
        print(model_name, str(model_accuracy))
        if model_accuracy > picked_accuracy:
            picked_model_name = model_name
            picked_accuracy, y_test, y_pred = model_accuracy, model_y_test, model_y_pred
    return picked_model_name, picked_accuracy


def get_train_test(games_df, features):
    games_df = games_df.sort_values('match_date')
    split_index = int(0.75 * len(games_df))
    train_df = games_df[:split_index]
    test_df = games_df[split_index:]
    return train_df[features], test_df[features], train_df['player_1_won'], test_df['player_1_won']
