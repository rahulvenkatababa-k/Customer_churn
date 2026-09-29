from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    ingested_data_path: str
    dataset_version: str
