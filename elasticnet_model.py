

import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import model_selection
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import learning_curve


import mlflow
import mlflow.sklearn

model_name = 'ElasticNet_model'

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


    alphas = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():

       
       #learning curves
        train_sizes = range(1,272, 5)
        train_sizes, train_scores, validation_scores = learning_curve(
        estimator = ElasticNet(alpha=alphas, l1_ratio=l1_ratio, random_state=42),
        X = X_train,
        y = y_train, 
        train_sizes = train_sizes,
        cv = 5,
        scoring = 'neg_mean_squared_error')
        train_scores_mean = -train_scores.mean(axis = 1)
        validation_scores_mean = -validation_scores.mean(axis = 1)





        #logging parameters
        mlflow.log_param('steps', train_sizes)
        mlflow.log_param("alphas", alphas)
        mlflow.log_param("l1_ratio", l1_ratio)

        #logging evolution of metrics, train loss and validation loss
        for epoch in range(0, len(train_sizes)):
            mlflow.log_metric(key="train_loss", value=train_scores_mean[epoch], step=train_sizes[epoch])
            mlflow.log_metric(key="val_loss", value=validation_scores_mean[epoch], step=train_sizes[epoch])
            lr = ElasticNet(alpha=alphas, l1_ratio=l1_ratio, random_state=42)
            lr.fit(X_train[:train_sizes[epoch]], y_train[:train_sizes[epoch]])
            predicted_qualities = lr.predict(X_validation)
            (rmse, mae, r2) = eval_metrics(y_test, predicted_qualities)
            mlflow.log_metric(key="rmse", value=rmse, step=train_sizes[epoch])
            mlflow.log_metric(key="mae", value=mae, step=train_sizes[epoch])
            mlflow.log_metric(key="r2", value=r2, step=train_sizes[epoch])
           
        
        mlflow.sklearn.log_model(lr, "model")
        mlflow.set_tag('model',model_name)
        
        #logging mrtrics with whole dataset
        lr = ElasticNet(alpha=alphas, l1_ratio=l1_ratio, random_state=42)
        lr.fit(X_train, y_train)
        predicted_qualities = lr.predict(X_validation)
        (rmse, mae, r2) = eval_metrics(y_test, predicted_qualities)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae) 
        
        #printing
        print(f"Elasticnet model (alpha={alphas}, l1_ratio={l1_ratio})")
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)
        print('completed logs')
    
#MLproject
# name: My Project

# conda_env: conda.yaml

# entry_points:
#   main:   
#     paramaters :
#       alpha: float
#       l1_ratio: {type: float, default: 0.1}
#     command: "python elasticnet_model.py" 