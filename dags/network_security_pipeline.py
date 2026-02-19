import os
import pickle
import pendulum
from airflow import DAG
from airflow.decorators import task
from network_security.components.data_ingestion import DataIngestion
from network_security.components.data_validation import DataValidation
from network_security.components.data_transformation import DataTransformation
from network_security.components.model_trainer import ModelTrainer
from network_security.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)
from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging
import sys

# ----------------------
# DAG configuration
# ----------------------
# Save artifacts and logs in current working directory
BASE_DIR = os.getcwd()
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
}

dag = DAG(
    dag_id='network_security_ml_pipeline_local',
    default_args=default_args,
    description='TaskFlow DAG saving artifacts/logs in local working directory',
    schedule=None,
    start_date=pendulum.now("UTC").subtract(days=1),
    catchup=False,
    max_active_runs=1
)

training_pipeline_config = TrainingPipelineConfig()

# ----------------------
# Helper functions
# ----------------------
def save_artifact(artifact, filename):
    path = os.path.join(ARTIFACT_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(artifact, f)
    return path

def load_artifact(filename):
    path = os.path.join(ARTIFACT_DIR, filename)
    with open(path, "rb") as f:
        return pickle.load(f)

# ----------------------
# TaskFlow tasks
# ----------------------
@task(dag=dag)
def data_ingestion_task():
    try:
        config = DataIngestionConfig(training_pipeline_config)
        ingestion = DataIngestion(config)
        logging.info("Initiating Data Ingestion")
        artifact = ingestion.initiate_data_ingestion()
        logging.info("Data Ingestion Completed")
        return save_artifact(artifact, "data_ingestion.pkl")
    except Exception as e:
        raise NetworkSecurityException(e, sys)

@task(dag=dag)
def data_validation_task(artifact_path: str):
    try:
        ingestion_artifact = load_artifact(artifact_path)
        config = DataValidationConfig(training_pipeline_config)
        validation = DataValidation(ingestion_artifact, config)
        logging.info("Initiating Data Validation")
        artifact = validation.initiate_data_validation()
        logging.info("Data Validation Completed")
        return save_artifact(artifact, "data_validation.pkl")
    except Exception as e:
        raise NetworkSecurityException(e, sys)

@task(dag=dag)
def data_transformation_task(artifact_path: str):
    try:
        validation_artifact = load_artifact(artifact_path)
        config = DataTransformationConfig(training_pipeline_config)
        logging.info("Data Transformation Started")
        transformation = DataTransformation(validation_artifact, config)
        artifact = transformation.initiate_data_transformation()
        logging.info("Data Transformation Completed")
        return save_artifact(artifact, "data_transformation.pkl")
    except Exception as e:
        raise NetworkSecurityException(e, sys)

@task(dag=dag)
def model_trainer_task(artifact_path: str):
    try:
        transformation_artifact = load_artifact(artifact_path)
        config = ModelTrainerConfig(training_pipeline_config)
        trainer = ModelTrainer(model_trainer_config=config, data_transformation_artifact=transformation_artifact)
        logging.info("Model Training Started")
        artifact = trainer.initiate_model_trainer()
        logging.info("Model Training Completed")
        return save_artifact(artifact, "model_trainer.pkl")
    except Exception as e:
        raise NetworkSecurityException(e, sys)

# ----------------------
# DAG dependencies
# ----------------------
ingest_path = data_ingestion_task()
validate_path = data_validation_task(ingest_path)
transform_path = data_transformation_task(validate_path)
train_path = model_trainer_task(transform_path)
