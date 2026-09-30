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

