import sys
import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from retainAI.entity.config_entity import ModelTrainerConfig
from retainAI.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact
)
from retainAI.exception.exception import RetainAIException
from retainAI.logging.logger import logging
from retainAI.utils.main_utils.utils import save_object, load_numpy_array_data


class ModelTrainer:
    def __init__(self,
                 data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def train_and_evaluate(self):
        try:
            # Load transformed data
            train_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            # Train XGBoost
            model = XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            model.fit(X_train, y_train)

            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # Metrics
            train_acc = accuracy_score(y_train, y_pred_train)
            test_acc = accuracy_score(y_test, y_pred_test)
            precision = precision_score(y_test, y_pred_test)
            recall = recall_score(y_test, y_pred_test)
            roc_auc = roc_auc_score(y_test, y_pred_test)

            logging.info(f"Train Accuracy: {train_acc:.4f}")
            logging.info(f"Test Accuracy:  {test_acc:.4f}")
            logging.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, ROC-AUC: {roc_auc:.4f}")

            # Save model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )

            artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_accuracy=train_acc,
                test_accuracy=test_acc,
                model_name="XGBoost"
            )
            return artifact

        except Exception as e:
            raise RetainAIException(e, sys)