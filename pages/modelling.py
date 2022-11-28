import streamlit as st
from sklearn.utils.validation import joblib
from sklearn.model_selection import KFold

def cross_validation(model, X, y):
    # prepare cross validation
    kf = KFold(n_splits=4)
    kf.get_n_splits(X)

    # enumerate splits
    i = 1

    score = 0
    for train_index, test_index in kf.split(X):
        i += 1
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        score_test = model.score(X_test, y_test)
        if score_test > score:
            X_train_best = train_index
            y_train_best = train_index
            score = score_test

    model = model.fit(X.iloc[X_train_best], y[y_train_best])
    return model.score(X_test, y_test)


def modelling():
    st.title("Modelling")

    st.radio(
        "Choose model",
        ("Gaussian Naive Bayes", "KNN", "Decision Tree"),
        key="model"
    )

    # create 3 output
    if "model" in  st.session_state:
        choose_model = st.session_state.model

        X = st.session_state.X
        y = st.session_state.y

        if choose_model == "Gaussian Naive Bayes":
            model = joblib.load("nb.joblib")
        elif choose_model == "KNN":
            model = joblib.load("knn.joblib")
        else:
            model = joblib.load("tree.joblib")

        score = cross_validation(model, X, y)
        st.success(f"Dengan model {choose_model}, di dapatkan akurasi sebesar: {score}", icon="✅")