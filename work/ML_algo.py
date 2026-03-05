import pickle
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
# --------------------------------
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
# ------------------------------
# Random Forest Classifier
def rf_param():
    rf_class_param_grid = {
    "n_estimators": [50, 100, 200, 300],        # number of trees
    "criterion": ["gini", "entropy", "log_loss"],  # quality of split
    "max_depth": [None, 5, 10, 20, 30],        # max depth of each tree
    "min_samples_split": [2, 5, 10],           # min samples required to split
    "min_samples_leaf": [1, 2, 4],             # min samples required at leaf
    "max_features": ["auto", "sqrt", "log2"],  # number of features to consider
    "bootstrap": [True, False]                 # whether to use bootstrap samples
}

# Random Forest Regressor
    rf_reg_param_grid = {
    "n_estimators": [50, 100, 200, 300],
    "criterion": ["squared_error", "absolute_error", "friedman_mse", "poisson"],
    "max_depth": [None, 5, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["auto", "sqrt", "log2"],
    "bootstrap": [True, False]
}
    return rf_reg_param_grid, rf_class_param_grid

# ----------------------------------

def preprocess_and_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    # transformers
    numeric_transformes = Pipeline(steps=[
        ("Imputation", SimpleImputer(strategy="median")),
        ("scaling", StandardScaler())
    ])

    categorical_transformers = Pipeline(steps=[
        ("Imputation", SimpleImputer(strategy="most_frequent")),
        ("Encoding", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("Number", numeric_transformes, numeric_features),
            ("Category", categorical_transformers, categorical_features)
        ]
    )

    return X_train, X_test, y_train, y_test, preprocess
# --------------------------------
def model_train(rf_param_grid, X, y, choice:str):
    X_train, X_test, y_train, y_test, preprocess = preprocess_and_split(X,y)
    if choice=="Regression":
        estimator = RandomForestRegressor(random_state=42)
        search = RandomizedSearchCV(
            estimator,
            rf_param_grid,
            n_iter=100,
            cv=5,
            scoring="r2",
            n_jobs=-1,
            random_state=42
        )
        pipeline = Pipeline(steps=[
    ("Preprocess",preprocess),
    ("Regression",search)
])
    else:
        estimator = RandomForestClassifier(random_state=42)
        search = RandomizedSearchCV(
            estimator,
            rf_param_grid,
            n_iter=100,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
            random_state=42
        )
        
        pipeline = Pipeline(steps=[
        ("Preprocess",preprocess),
        ("Classify",RandomizedSearchCV(RandomForestClassifier(),rf_param_grid,n_iter=100))
    ])
        
    model = pipeline.fit(X_train,y_train)
    preds = model.predict(X_test)
    if choice=="Regression":
        score = r2_score(y_test, preds)
    else:
        score = accuracy_score(y_test, preds)    
    # Saving machine learning model 
    with open("model.pkl","wb") as F:
        pickle.dump(model, F)
    message = "The Machine Learning model is trained on Your Data"
    return model, score, message
