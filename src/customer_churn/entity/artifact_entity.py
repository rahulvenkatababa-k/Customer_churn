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
    X_val_path: str
    y_val_path: str
    X_test_path: str
    y_test_path: str

@dataclass
class ModelTrainerArtifact:
    baseline_model_path: str
    candidate_model_paths: dict[str, str]
    experiment_results_path: str

@dataclass
class ModelEvaluationArtifact:
    evaluation_report_file_path: str
    confusion_matrix_file_paths : dict[str, str]
    registered_model_uri: str

@dataclass
class ModelPusherArtifact:
    model_name: str
    model_alias: str
    model_version: str
