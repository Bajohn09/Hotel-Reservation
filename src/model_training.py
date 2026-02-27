import os
import pandas as pd
import joblib

from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from src.logger import get_logger
from config.paths_config import *
from config.model_params import *
from utils.common_functions import read_yaml, load_data
from scipy.stats import randint

import mlflow

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self, train_path, test_path, model_output_path):

        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path

        self.params_list = LIGHTGBM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS

    def load_and_split_data(self):

        try:
            logger.info(f"Loading Data from {self.train_path}")
            train_df = load_data(self.train_path)
            
            logger.info(f"Loading Data from {self.test_path}")
            test_df = load_data(self.test_path)
            
            # Are the columns consistent?
            X_train = train_df.drop(columns=["booking_status"])
            y_train = train_df["booking_status"]

            X_test = test_df.drop(columns=["booking_status"])
            y_test = test_df["booking_status"]

            logger.info(f"Data splitted successfully.")

            return X_train, y_train, X_test, y_test
        
        except Exception as e:
            logger.error(f"Error during loading data {e}")
            raise e
    
    def train_lgbm(self, X_train, y_train):
        try:
            logger.info("Initializing model")
            lgbm_model = lgb.LGBMClassifier(random_state=42)

            logger.info("Starting our hyperparameter tuning")

            random_search = RandomizedSearchCV(
                estimator=lgbm_model, # type: ignore
                param_distributions=self.params_list,
                **self.random_search_params                
            )

            logger.info("Starting our model training.")

            random_search.fit(X_train, y_train)

            logger.info("Hyperparameter tuning completed.")

            best_params = random_search.best_params_
            best_lgbm_model = random_search.best_estimator_

            logger.info(f"Best Parameters are: {best_params}")

            return best_lgbm_model

        except Exception as e:
            logger.error(f"Error while training the model {e}")
            raise e
    
    def evaluate_model(self, model, X_test, y_test):
        
        metrics = {}
        try:
            logger.info("Model Evaluation...")

            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            logger.info(f"Accuracy: {accuracy}")
            logger.info(f"Precision Score: {precision}")
            logger.info(f"Recall Score: {recall}")
            logger.info(f"F1 Score: {f1}")

            
            metrics["Accuracy"] = accuracy
            metrics["Precision"] = precision
            metrics["Recall"] = recall
            metrics["F1-Score"] = f1

            return metrics

        except Exception as e:
            logger.error(f"Error while model Evaluation {e}")
            raise e
        
    def save_model(self, model):

        try:
            if not os.path.exists(os.path.dirname(self.model_output_path)):
                os.makedirs(os.path.dirname(self.model_output_path))
            
            logger.info("Saving the model")
            joblib.dump(model, self.model_output_path)
            logger.info(f"Save model successfully to: {self.model_output_path}")
        
        except Exception as e:
            logger.error(f"Error while saving the model {e}")
            raise e
    
    def run(self):

        try:
            with mlflow.start_run():
                logger.info("Starting model training pipeline")
                logger.info("Starting MLFlow Experimentation")

                mlflow.log_artifact(self.train_path, artifact_path="datasets")
                mlflow.log_artifact(self.test_path, artifact_path="datasets")

                X_train, y_train, X_test, y_test = self.load_and_split_data()
                best_lgbm_model = self.train_lgbm(X_train, y_train)
                metrics = self.evaluate_model(best_lgbm_model, X_test, y_test)
                self.save_model(best_lgbm_model)

                # Logging experiment data
                mlflow.log_artifact(self.model_output_path)
                mlflow.log_params(best_lgbm_model.get_params())
                mlflow.log_metrics(metrics)

                logger.info("Model training successfully completed.")

                
        except Exception as e:
            logger.error(f"Error while saving the model {e}")
            raise e




        


