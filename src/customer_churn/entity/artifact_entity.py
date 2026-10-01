from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    ingested_data_path: str
    dataset_version: str

@dataclass
class DataValidationArtifact:
    validated_status: bool
    validated_data_path: str | None
    validation_report_path: str

@dataclass
class DataTransformationArtifact:
    transformed_data_path: str

@dataclass
class FeatureEngineeringArtifact:
    X_train_path: str
    y_train_path: str
    X_test_path: str
    y_test_path: str
    preprocessor_file_path: str

@dataclass
class ModelTrainerArtifact:
    model_path : str