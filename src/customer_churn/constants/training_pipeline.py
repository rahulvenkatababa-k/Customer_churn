#Generic Constants

PIPELINE_NAME = "customer_churn"
ARTIFACTS_DIR_NAME = 'artifacts'
RANDOM_STATE = 42

#data ingestion stage related constants start with var name DATA_INGESTION

DATA_INGESTION_DIR_NAME = "data_ingestion"
DATA_INGESTION_INGESTED_DIR_NAME = "ingested"

#data validation stage related constants start with var name DATA_VALIDATION

DATA_VALIDATION_DIR_NAME = "data_validation"
DATA_VALIDATION_VALID_DIR_NAME = "validated"
DATA_VALIDATION_VALIDATED_DATAFRAME_NAME = "validated_customer_churn.csv"
DATA_VALIDATION_REPORT_FILE_NAME = "validation_report.json"

#data transformation stage related constants start with var name DATA_TRANSFORMATION

DATA_TRANSFORMATION_DIR_NAME = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DIR_NAME = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_DATAFRAME_NAME = "transformed_customer_churn.csv"

#feature engineering stage related constants start with var name FEATURE_ENGINEERING

FEATURE_ENGINEERING_DIR_NAME = "feature_engineering"
FEATURE_ENGINEERING_ENGINEERED_DIR_NAME = "engineered"
FEATURE_ENGINEERING_PREPROCESSOR_FILE_NAME = 'preprocessor.pkl'
FEATURE_ENGINEERING_X_TRAIN_FILE_NAME = "X_train.csv"
FEATURE_ENGINEERING_X_TEST_FILE_NAME = "X_test.csv"
FEATURE_ENGINEERING_Y_TRAIN_FILE_NAME = "y_train.csv"
FEATURE_ENGINEERING_Y_TEST_FILE_NAME = "y_test.csv"

#Model Trainer stage related constants start with var name MODEL_TRAINER

MODEL_TRAINER_DIR_NAME = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR_NAME = "trained_model"
MODEL_TRAINER_FILE_NAME = "model.pkl"

