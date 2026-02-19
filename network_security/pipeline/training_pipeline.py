import os
import sys

from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging

from network_security.components.data_ingestion import DataIngestion
from network_security.components.data_validation import DataValidation
from network_security.components.data_transformation import DataTransformation
from network_security.components.model_trainer import ModelTrainer

from network_security.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)

from network_security.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)

from network_security.cloud.s3_syncer import S3Sync


class TrainingPipeline:
    def __init__(self):
        try:
            # 🔥 Get timestamp from Airflow ENV
            timestamp = os.getenv("PIPELINE_TIMESTAMP")

            if not timestamp:
                raise ValueError(
                    "PIPELINE_TIMESTAMP not found. Ensure Airflow passes it."
                )

            self.training_pipeline_config = TrainingPipelineConfig(
                timestamp=timestamp
            )

            logging.info(f"Pipeline initialized with timestamp: {timestamp}")
            logging.info(
                f"Artifact directory: {self.training_pipeline_config.artifact_dir}"
            )

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    # ----------------------------
    # Full pipeline methods (kept for future use)
    # ----------------------------

    def start_data_ingestion(self):
        try:
            data_ingestion_config = DataIngestionConfig(
                training_pipeline_config=self.training_pipeline_config
            )

            logging.info("Starting Data Ingestion")

            data_ingestion = DataIngestion(
                data_ingestion_config=data_ingestion_config
            )

            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

            return data_ingestion_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def start_data_validation(
        self, data_ingestion_artifact: DataIngestionArtifact
    ):
        try:
            data_validation_config = DataValidationConfig(
                training_pipeline_config=self.training_pipeline_config
            )

            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=data_validation_config,
            )

            data_validation_artifact = data_validation.initiate_data_validation()

            return data_validation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def start_data_transformation(
        self, data_validation_artifact: DataValidationArtifact
    ):
        try:
            data_transformation_config = DataTransformationConfig(
                training_pipeline_config=self.training_pipeline_config
            )

            data_transformation = DataTransformation(
                data_validation_artifact=data_validation_artifact,
                data_transformation_config=data_transformation_config,
            )

            data_transformation_artifact = (
                data_transformation.initiate_data_transformation()
            )

            return data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def start_model_trainer(
        self, data_transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:
        try:
            model_trainer_config = ModelTrainerConfig(
                training_pipeline_config=self.training_pipeline_config
            )

            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=model_trainer_config,
            )

            model_trainer_artifact = model_trainer.initiate_model_trainer()

            logging.info("Model Training Completed Successfully")

            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)


# ==========================================================
# 🔥 TRAINING STAGE ENTRYPOINT (Used by Airflow BashOperator)
# ==========================================================

if __name__ == "__main__":

    print("🚀 Starting Training Stage Only")

    try:
        pipeline = TrainingPipeline()

        artifact_dir = pipeline.training_pipeline_config.artifact_dir

        # 🔥 Build transformation artifact paths
        transformed_train_path = os.path.join(
            artifact_dir,
            "data_transformation",
            "transformed",
            "train.npy",
        )

        transformed_test_path = os.path.join(
            artifact_dir,
            "data_transformation",
            "transformed",
            "test.npy",
        )

        transformed_object_path = os.path.join(
            artifact_dir,
            "data_transformation",
            "transformed_object",
            "preprocessor.pkl",
        )

        # 🔥 Safety check (VERY IMPORTANT in Docker)
        for path in [
            transformed_train_path,
            transformed_test_path,
            transformed_object_path,
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Required transformation file not found: {path}"
                )

        logging.info("All transformation files verified successfully")

        data_transformation_artifact = DataTransformationArtifact(
            transformed_train_file_path=transformed_train_path,
            transformed_test_file_path=transformed_test_path,
            transformed_object_file_path=transformed_object_path,
        )

        # 🚀 Run training
        pipeline.start_model_trainer(data_transformation_artifact)

        print("✅ Training Completed Successfully")

    except Exception as e:
        logging.error("❌ Training Failed")
        raise NetworkSecurityException(e, sys)