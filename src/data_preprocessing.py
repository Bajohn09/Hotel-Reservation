import os
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml, load_data

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

logger = get_logger(__name__)

class DataProcessor():
    def __init__(self, train_path, test_path, processed_dir, config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir
        
        self.config = read_yaml(config_path)

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def preprocess_data(self, df):
        try:
            logger.info("Starting data preprocessing")

            logger.info("Dropping useless columns")
            df.drop(columns=["Unnamed: 0", "Booking_ID"], inplace=True)

            logger.info("Dropping duplicates")
            df.drop_duplicates(inplace=True)

            cat_cols = self.config["data_processing"]["categorical_columns"]
            num_cols = self.config["data_processing"]["numerical_columns"]

            logger.info("Applying Label Encoder")
            label_encoder = LabelEncoder()
            
            # Mappings are important since they give us information on how to transform back the data to their original form.
            mappings = {}

            for col in cat_cols:
                df[col] = label_encoder.fit_transform(df[col])
                
                mappings[col] = {label: code for label, code in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))} # type: ignore

            logger.info("Label Mappings Are")
            for col, mapping in mappings.items():
                logger.info(f"{col}: {mapping}")

            logger.info("Handle skew data")
            skew_limit = self.config["data_processing"]["skewness_threshold"]
            
            skewness = df[num_cols].apply(lambda x: x.skew())

            for col in skewness[skewness>skew_limit].index:
                df[col] = np.log1p(df[col]) # Apply log transformation to the skewed features.

            return df
        
        except Exception as e:
            logger.error(f"Error during preprocessing {e}")
            raise CustomException("Error while preprocess data", e)
        
    def balance_data(self, df):
        try:
            logger.info("Handling Imbalanced Data")
            X = df.drop(columns="booking_status")
            y = df["booking_status"]

            smote = SMOTE(random_state=42)
            x_res, y_res = smote.fit_resample(X, y)  # type: ignore

            balanced_df = pd.DataFrame(x_res, columns=X.columns)
            balanced_df["booking_status"] = y_res

            logger.info("Data balanced succesfully")

            return balanced_df

        except Exception as e:
            logger.error(f"Error during balancing data {e}")
            raise CustomException("Error while balancing data", e)
        
    def select_features(self, df):
        try:
            logger.info("Starting feature selection.")
            
            X = df.drop(columns="booking_status")
            y = df["booking_status"]

            model = RandomForestClassifier(random_state=42)
            model.fit(X,y)

            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
                })
            
            top_feature_importance_df = (
                feature_importance.sort_values(by="importance", ascending=False)
            )
    
            num_features_to_select = self.config["data_processing"]["num_features_to_select"]

            top_10_features = top_feature_importance_df.iloc[:num_features_to_select,:]

            top_10_df = df.loc[:, list(top_10_features["feature"].values) + ["booking_status"]]

            logger.info("Feature Selection completed.")
            logger.info(f"Top 10 features: {list(top_10_features['feature'].values)}")

            return top_10_df
        
        except Exception as e:
            logger.error(f"Error during feature selection step {e}")
            raise CustomException("Error while feature selection", e)
    

    def save_data(self, df, file_path):
        try:
            logger.info("Starting feature selection.")
            df.to_csv(file_path, index=False)
            logger.info("Data saved successfully.")
        except Exception as e:
            logger.error(f"Error during saving data {e}")
            raise CustomException("Error while saving daat", e)
        
    def process(self):

        try:
            logger.info("Loading data from Raw Directory.")

            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            train_df = self.preprocess_data(train_df)
            test_df = self.preprocess_data(test_df)

            train_df = self.balance_data(train_df)
            
            train_df = self.select_features(train_df)
            test_df = test_df.loc[:, train_df.columns]

            self.save_data(train_df, PROCESSED_TRAIN_DATA_PATH)
            self.save_data(test_df, PROCESSED_TEST_DATA_PATH)

            logger.info("Data processing completed successfully.")

        except Exception as e:
            logger.error(f"Error during preprocessing pipeline {e}")
            raise CustomException("Error while preprocessing pipeline", e)




        





            

