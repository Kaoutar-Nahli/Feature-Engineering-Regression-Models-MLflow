import os
from uuid import RFC_4122
import warnings
import sys

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn import  model_selection
from sklearn.model_selection import learning_curve

import mlflow
import mlflow.sklearn

model_name = 'Random_forest_model'

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2



if __name__ == "__main__":
       warnings.filterwarnings("ignore")
       np.random.seed(40)

       
       #Reading the data
       df = pd.read_csv('material_rating_reviews_2021091618.csv')
       y = df['combined_price']
       X =df[['rating', 'number_reviews', 'material_cork',
              'material_cotton', 'material_foam', 'material_jute', 'material_nbr',
              'material_neoprene', 'material_per', 'material_polyester',
              'material_pu', 'material_pvc', 'material_rubber', 'material_suede',
              'material_tpe']]
       print(X.shape)

       #splitting data into training, validation and test
       X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size=0.4, stratify= X[['material_cork',
              'material_cotton', 'material_foam', 'material_jute', 'material_nbr',
              'material_neoprene', 'material_per', 'material_polyester',
              'material_pu', 'material_pvc', 'material_rubber', 'material_suede',
              'material_tpe']])

       X_validation, X_test, y_validation, y_test = model_selection.train_test_split(
              X_test, y_test, test_size=0.5)
              
       print(X_train.shape, X_validation.shape, X_test.shape)

       n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 0.5
       min_samples_leaf = int(sys.argv[2]) if len(sys.argv) > 2 else 0.5
             

       with mlflow.start_run():

       
              #learning curves
              train_sizes = range(1,272, 5)
              train_sizes, train_scores, validation_scores = learning_curve(
              estimator = RandomForestRegressor(n_estimators=n_estimators , min_samples_leaf=min_samples_leaf, random_state=42),
              X = X_train,
              y = y_train, 
              train_sizes = train_sizes,
              cv = 5,
              scoring = 'neg_mean_squared_error')
              train_scores_mean = -train_scores.mean(axis = 1)
              validation_scores_mean = -validation_scores.mean(axis = 1)





              #logging parameters
              mlflow.log_param('steps', train_sizes)
              mlflow.log_param("n_estimators", n_estimators)
              mlflow.log_param("min_samples_leaf", min_samples_leaf)

              #logging evolution of metrics, train loss and validation loss
              for epoch in range(0, len(train_sizes)):
                     mlflow.log_metric(key="train_loss", value=train_scores_mean[epoch], step=train_sizes[epoch])
                     mlflow.log_metric(key="val_loss", value=validation_scores_mean[epoch], step=train_sizes[epoch])
                     rf= RandomForestRegressor(n_estimators=n_estimators , min_samples_leaf=min_samples_leaf, random_state=42)
                     rf.fit(X_train[:train_sizes[epoch]], y_train[:train_sizes[epoch]])
                     predicted_qualities = rf.predict(X_validation)
                     (rmse, mae, r2) = eval_metrics(y_test, predicted_qualities)
                     mlflow.log_metric(key="rmse", value=rmse, step=train_sizes[epoch])
                     mlflow.log_metric(key="mae", value=mae, step=train_sizes[epoch])
                     mlflow.log_metric(key="r2", value=r2, step=train_sizes[epoch])

       
              mlflow.sklearn.log_model(rf, "model")
              mlflow.set_tag('model',model_name)

              #logging mrtrics with whole dataset
              rf = RandomForestRegressor(n_estimators=n_estimators , min_samples_leaf=min_samples_leaf, random_state=42)
              rf.fit(X_train, y_train)
              predicted_qualities = rf.predict(X_validation)
              (rmse, mae, r2) = eval_metrics(y_test, predicted_qualities)
              mlflow.log_metric("rmse", rmse)
              mlflow.log_metric("r2", r2)
              mlflow.log_metric("mae", mae) 

              #printing
              print("Random forest model (n_estimators=%f, min_samples_leaf=%f):" % (n_estimators, min_samples_leaf))
              print("  RMSE: %s" % rmse)
              print("  MAE: %s" % mae)
              print("  R2: %s" % r2)
              print('completed logs')

    


              # # THIS IS HOW YOU PARSE ARGUMENTS FROM THE COMMAND LINE
              # def get_flags_passed_in_from_terminal():
              #        parser = argparse.ArgumentParser()
              #        parser.add_argument('-r')
              #        args = parser.parse_args()
              #        return args
              # args = get_flags_passed_in_from_terminal()
              # print(args)
#MLproject
# name: My Project

# conda_env: conda.yaml

# entry_points:
#   main:
#     paramaters:  
#       n_estimators: float
#       min_samples_leaf: {type: float, default: 5}
#     command: "python random_forest_model.py {n_estimators} {min_samples_leaf}" 