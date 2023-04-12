import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


def normalization(df, df_min, df_max):
    return (df - df_min) / (df_max - df_min)


if __name__ == '__main__':
    df = pd.read_csv('to_model_train.csv', index_col=0)

    y = df['code']
    X = df.drop(['code', 'uid', 'keywords', 'content_url'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    max_v, min_v = X_train.max(), X_train.min()
    X_train_norm = normalization(X_train, min_v, max_v)
    X_test_norm = normalization(X_test, min_v, max_v)

    model = RandomForestClassifier(n_estimators=1000,
                                   criterion="entropy",
                                   random_state=42,
                                   max_depth=6,
                                   #class_weight=
                                   )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    model = CatBoostClassifier(iterations=500,
                               learning_rate=0.05,
                               depth=6,
                               l2_leaf_reg=0,
                               loss_function='CrossEntropy',
                               # use_best_model=True,
                               text_features=['content']
                               )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    predictions = [round(value) for value in pred]

    # # evaluate predictions
    acc = accuracy_score(y_test, pred)
    print("Accuracy: %.2f%%" % (acc * 100.0))
