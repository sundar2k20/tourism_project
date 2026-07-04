
# for data manipulation
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# for model serialization
import joblib
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError


repo_id = "sundar2k20/tourism_project"
api = HfApi(token=os.getenv("HF_TOKEN"))

Xtrain_path = api.hf_hub_download(repo_id=repo_id, filename="Xtrain.csv", repo_type="dataset")
Xtest_path = api.hf_hub_download(repo_id=repo_id, filename="Xtest.csv", repo_type="dataset")
ytrain_path = api.hf_hub_download(repo_id=repo_id, filename="ytrain.csv", repo_type="dataset")
ytest_path = api.hf_hub_download(repo_id=repo_id, filename="ytest.csv", repo_type="dataset")

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

# Define features
numeric_features = [
    'Age',     # Customer's age
    'CityTier', # The city category based on development, population, and living standards (Tier 1 > Tier 2 > Tier 3)
    'NumberOfPersonVisiting', # Total number of people accompanying the customer on the trip
    'PreferredPropertyStar',  # Preferred hotel rating by the customer
    'NumberOfTrips',     # Average number of trips the customer takes annually
    'NumberOfChildrenVisiting', # Number of children below age 5 accompanying the customer
    'MonthlyIncome', # Gross monthly income of the customer
    'PitchSatisfactionScore', # Score indicating the customer's satisfaction with the sales pitch
    'NumberOfFollowups', # Total number of follow-ups by the salesperson after the sales pitch
    'DurationOfPitch' # Duration of the sales pitch delivered to the customer
]
categorical_features = [
    'TypeofContact', # The method by which the customer was contacted (Company Invited or Self Inquiry)
    'Occupation', # Customer's occupation (e.g., Salaried, Freelancer)
    'Gender', # Gender of the customer (Male, Female)
    'MaritalStatus', # Marital status of the customer (Single, Married, Divorced)
    'Designation', # Customer's designation in their current organization
    'ProductPitched', # The type of product pitched to the customer
    'Passport', # Whether the customer holds a valid passport (0: No, 1: Yes)
    'OwnCar' # Whether the customer owns a car (0: No, 1: Yes)
]

# Preprocessing pipeline
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define XGBoost Classifier
xgb_model = xgb.XGBClassifier(random_state=42, objective="binary:logistic")

# Define hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 100, 200],
    'xgbclassifier__max_depth': [3, 5, 7],
    'xgbclassifier__learning_rate': [0.01, 0.1, 0.2],
    'xgbclassifier__colsample_bytree': [0.6, 0.8, 1.0],
    'xgbclassifier__subsample': [0.6, 0.8, 1.0],
    'xgbclassifier__gamma': [0, 0.1, 0.2],
    'xgbclassifier__reg_lambda': [0.1, 1, 10],
}

# Create pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Grid search with cross-validation
grid_search = GridSearchCV(
    model_pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1
)
grid_search.fit(Xtrain, ytrain)

# Best model
best_model = grid_search.best_estimator_
print("Best Params:\n", grid_search.best_params_)

# Predictions
y_pred_train = best_model.predict(Xtrain)
y_pred_test = best_model.predict(Xtest)

# Evaluation
print("\nTraining Performance:")
print("Accuracy:", accuracy_score(ytrain, y_pred_train))
print("Precision:", precision_score(ytrain, y_pred_train))
print("Recall:", recall_score(ytrain, y_pred_train))
print("F1 Score:", f1_score(ytrain, y_pred_train))

print("\nTest Performance:")
print("Accuracy:", accuracy_score(ytest, y_pred_test))
print("Precision:", precision_score(ytest, y_pred_test))
print("Recall:", recall_score(ytest, y_pred_test))
print("F1 Score:", f1_score(ytest, y_pred_test))

# Save best model
joblib.dump(best_model, "best_tourism_model_v1.joblib")


# Upload to Hugging Face
repo_id = "sundar2k20/tourism_project_model"
repo_type = "model"

api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Model Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Model Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Model Space '{repo_id}' created.")

api.upload_file(
    path_or_fileobj="best_tourism_model_v1.joblib",
    path_in_repo="best_tourism_model_v1.joblib",
    repo_id=repo_id,
    repo_type=repo_type,
)
