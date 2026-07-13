import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek

from retainAI.entity.config_entity import DataTransformationConfig
from retainAI.entity.artifact_entity import (
    DataValidationArtifact,
    DataTransformationArtifact
)
from retainAI.exception.exception import RetainAIException
from retainAI.logging.logger import logging
from retainAI.constants.training_pipeline import TARGET_COLUMN
from retainAI.utils.main_utils.utils import save_numpy_array_data, save_object


class DataTransformation:
    def __init__(self,
                 data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        self.data_validation_artifact = data_validation_artifact
        self.data_transformation_config = data_transformation_config

    @staticmethod
    def _read_csv(file_path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path)
            logging.info(f"Read {df.shape[0]} rows from {file_path}")
            return df
        except Exception as e:
            raise RetainAIException(e, sys)

    def _get_preprocessor(self, numerical_cols, categorical_cols):
        try:
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler())
            ])
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            preprocessor = ColumnTransformer(transformers=[
                ('num', num_pipeline, numerical_cols),
                ('cat', cat_pipeline, categorical_cols)
            ])
            logging.info("Preprocessor created: Numerical (median+RobustScaler), "
                         "Categorical (most_frequent+OneHotEncoder)")
            return preprocessor
        except Exception as e:
            raise RetainAIException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            train_df = self._read_csv(
                self.data_validation_artifact.valid_train_file_path
            )
            test_df = self._read_csv(
                self.data_validation_artifact.valid_test_file_path
)

            # 2. Drop rows with missing target (if any)
            train_df.dropna(subset=[TARGET_COLUMN], inplace=True)
            test_df.dropna(subset=[TARGET_COLUMN], inplace=True)

            # 3. Separate features and target
            X_train = train_df.drop(columns=[TARGET_COLUMN])
            y_train = train_df[TARGET_COLUMN]
            X_test = test_df.drop(columns=[TARGET_COLUMN])
            y_test = test_df[TARGET_COLUMN]

            # 4. Define column types (skip ID‑like columns)
            id_columns = ['EmployeeNumber']
            # Ordinal columns that should be treated as categorical for safety
            ordinal_cols = [
                'Education', 'EnvironmentSatisfaction', 'JobInvolvement',
                'JobLevel', 'JobSatisfaction', 'PerformanceRating',
                'RelationshipSatisfaction', 'StockOptionLevel',
                'WorkLifeBalance'
            ]

            numerical_columns = [
                col for col in X_train.columns
                if X_train[col].dtype in ['int64', 'float64']
                and col not in id_columns
                and col not in ordinal_cols
            ]
            categorical_columns = [
                col for col in X_train.columns
                if X_train[col].dtype == 'object' or col in ordinal_cols
            ]

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            # 5. Fit preprocessor on training data
            preprocessor = self._get_preprocessor(numerical_columns, categorical_columns)
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            # 6. Map target: Yes->1, No->0
            y_train = y_train.map({'Yes': 1, 'No': 0})
            y_test = y_test.map({'Yes': 1, 'No': 0})

            # 7. SMOTE‑Tomek on training data only
            smt = SMOTETomek(random_state=42)
            X_train_final, y_train_final = smt.fit_resample(X_train_transformed, y_train)
            logging.info(
                f"After SMOTE-Tomek: X_train shape = {X_train_final.shape}, "
                f"y_train distribution = {np.bincount(y_train_final.astype(int))}"
            )

            # 8. Concatenate features + target into final arrays
            train_arr = np.c_[X_train_final, y_train_final]
            test_arr = np.c_[X_test_transformed, y_test]

            # 9. Save objects
            save_object(
                file_path=self.data_transformation_config.transformed_object_file_path,
                obj=preprocessor
            )
            save_numpy_array_data(
                file_path=self.data_transformation_config.transformed_train_file_path,
                array=train_arr
            )
            save_numpy_array_data(
                file_path=self.data_transformation_config.transformed_test_file_path,
                array=test_arr
            )

            # 10. Create and return artifact
            transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path
            )
            logging.info(f"Data Transformation artifact: {transformation_artifact}")
            return transformation_artifact

        except Exception as e:
            raise RetainAIException(e, sys)