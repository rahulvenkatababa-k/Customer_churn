import sys
from pathlib import Path


TEXT_TEMPLATES = {
    "config/config.yaml": "# Project configuration\n",
    ".env.example": """# Copy this file to .env and fill in local values.\n# Never commit .env.\n\nPOSTGRES_SQL_URI=\nAWS_REGION=\nMODEL_ARTIFACT_BUCKET=\nMODEL_NAME=NetworkSecurityModel\nMODEL_VERSION=\nMLFLOW_TRACKING_URI=\nMLFLOW_TRACKING_USERNAME=\nMLFLOW_TRACKING_PASSWORD=\nPREFECT_API_URL=http://127.0.0.1:4200/api\n""",
    ".github/workflows/ci.yml": """name: CI\n\non:\n  push:\n    branches: [main]\n  pull_request:\n    branches: [main]\n\njobs:\n  test:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v4\n      - uses: actions/setup-python@v5\n        with:\n          python-version: \"3.12\"\n      - run: python -m pip install --upgrade pip\n      - run: pip install -r requirements.txt\n      - run: pytest -v tests/unit\n      - run: docker build -t network-security-api:test .\n""",
    ".gitignore": """.venv/\nvenv/\n.env\n__pycache__/\n*.py[cod]\n*.egg-info/\n.pytest_cache/\n.ipynb_checkpoints/\n.vscode/\n.idea/\n*.log\nlogs/*\n!logs/.gitkeep\ndata/raw/*\ndata/interim/*\ndata/processed/*\n!data/raw/.gitkeep\n!data/interim/.gitkeep\n!data/processed/.gitkeep\nartifacts/*\n!artifacts/.gitkeep\nmlruns/*\n!mlruns/.gitkeep\nterraform/.terraform/\nterraform/*.tfstate\nterraform/*.tfstate.*\n""",
    ".dockerignore": """.git\n.github\n.venv\n__pycache__\n.pytest_cache\nnotebooks\ndata\nlogs\nartifacts\nmlruns\nterraform\n.env\n""",
    ".dvcignore": """.git\n.venv\n__pycache__\n""",
    ".prefectignore": """.git\n.venv\n__pycache__\nartifacts\nlogs\nmlruns\n""",
    "pyproject.toml": """[build-system]\nrequires = [\"setuptools>=61\"]\nbuild-backend = \"setuptools.build_meta\"\n\n[project]\nname = \"{package_name}\"\nversion = \"0.0.1\"\ndescription = \"End-to-end Network Security MLOps project\"\nrequires-python = \">=3.9\"\n\n[tool.setuptools.packages.find]\nwhere = [\"src\"]\n\n[tool.pytest.ini_options]\npythonpath = [\".\"]\ntestpaths = [\"tests\"]\n""",
    "requirements.txt": """pandas\nnumpy\nscikit-learn\npyyaml\npython-dotenv\nsetuptools\nmlflow\nboto3\npytest\nSQLAlchemy\npsycopg2-binary\nfeast\npandera\npsycopg[binary]\npsycopg_pool\ndvc\nprefect\nbotocore[crt]\n""",
    "requirements-api.txt": """fastapi\nuvicorn\npandas\nnumpy\nscikit-learn\njoblib\nprometheus-fastapi-instrumentator\nboto3\n""",
    "Dockerfile": """FROM python:3.12-slim\n\nWORKDIR /app\n\nCOPY requirements-api.txt .\nRUN pip install --no-cache-dir --upgrade pip && \\\n    pip install --no-cache-dir -r requirements-api.txt && \\\n    pip install --no-cache-dir --upgrade \"msgpack>=1.2.1\"\n\nCOPY src/ ./src/\nCOPY deployment/ ./deployment/\n\nEXPOSE 8000\nCMD [\"uvicorn\", \"src.networksecurity.app:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]\n""",
    "docker-compose.yml": """services:\n  network-security-api:\n    build:\n      context: .\n      dockerfile: Dockerfile\n    image: network-security-api:latest\n    container_name: network-security-api\n    ports:\n      - \"8000:8000\"\n    restart: unless-stopped\n""",
    "README.md": """# Network Security MLOps\n\nSee the project README for the full architecture, setup, development, CI, MLOps, API, Docker, Kubernetes, Terraform, and deployment documentation.\n""",
    "main.py": """from src.networksecurity.pipeline.prefect_pipeline import network_security_training_flow\n\n\nif __name__ == \"__main__\":\n    network_security_training_flow()\n""",
    "feature_repo/feature_store.yaml": """project: NetworkSecurity\nregistry:\n  path: data/registry.db\nprovider: local\n\nonline_store:\n  type: sqlite\n  path: data/online_store.db\n\noffline_store:\n  type: postgres\n  host: localhost\n  port: 5432\n  database: NetworkSecurity\n  db_schema: public\n  user: postgres\n  password: <SET_VIA_ENV_OR_LOCAL_CONFIG>\n  sslmode: disable\n""",
    "feast_example/feature_definitions.py": """# Optional Feast examples.\n# Keep project-specific feature definitions under feature_repo/.\n""",
    "k8s/configmap.yaml": """apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: network-security-api\n""",
    "k8s/deployment.yaml": """apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: network-security-api\nspec:\n  replicas: 1\n  selector:\n    matchLabels:\n      app: network-security-api\n  template:\n    metadata:\n      labels:\n        app: network-security-api\n    spec:\n      containers:\n        - name: network-security-api\n          image: network-security-api:latest\n          ports:\n            - containerPort: 8000\n""",
    "k8s/service.yaml": """apiVersion: v1\nkind: Service\nmetadata:\n  name: network-security-api\nspec:\n  selector:\n    app: network-security-api\n  ports:\n    - port: 8000\n      targetPort: 8000\n""",
    "k8s/secrets.yaml": """apiVersion: v1\nkind: Secret\nmetadata:\n  name: network-security-api\ntype: Opaque\nstringData: {}\n""",
    "monitoring/kube-prometheus-values.yaml": "# Prometheus/Grafana values go here.\n",
    "monitoring/servicemonitor.yaml": "# Prometheus ServiceMonitor goes here.\n",
    "alloy-values.yaml": "# Grafana Alloy values.\n",
    "alloy-rendered.yaml": "# Rendered Alloy manifest placeholder.\n",
    "loki-values.yaml": "# Loki values.\n",
    "loki-rendered.yaml": "# Rendered Loki manifest placeholder.\n",
}


STRUCTURAL_FILES = [
    ".github/workflows/cd.yml.deactivate",
    "prefect.yaml",
    "feast_example/feature_definitions.py",
    "helm/network-security-api/Chart.yaml",
    "helm/network-security-api/values.yaml",
    "helm/network-security-api/templates/_helpers.tpl",
    "helm/network-security-api/templates/configmap.yaml",
    "helm/network-security-api/templates/deployment.yaml",
    "helm/network-security-api/templates/secret.yaml",
    "helm/network-security-api/templates/service.yaml",
    "terraform/providers.tf",
    "terraform/main.tf",
    "terraform/locals.tf",
    "terraform/variables.tf",
    "terraform/outputs.tf",
    "terraform/modules/project_info/main.tf",
    "terraform/modules/project_info/variables.tf",
    "terraform/modules/project_info/outputs.tf",
    "terraform/modules/deployment/main.tf",
    "terraform/modules/deployment/variables.tf",
    "terraform/modules/deployment/outputs.tf",
    "terraform/modules/storage/main.tf",
    "terraform/modules/storage/variables.tf",
    "terraform/modules/storage/outputs.tf",
    "terraform/modules/container_registry/main.tf",
    "terraform/modules/container_registry/variables.tf",
    "terraform/modules/container_registry/outputs.tf",
    "terraform/modules/iam/main.tf",
    "terraform/modules/iam/variables.tf",
    "terraform/modules/iam/outputs.tf",
    "terraform/modules/iam/ecr-policy.tf",
    "terraform/modules/iam/eks-policy.tf",
    "terraform/modules/iam/eks_access.tf",
]


DIRECTORIES = [
    "src/{pkg}",
    "src/{pkg}/constants",
    "src/{pkg}/entity",
    "src/{pkg}/components",
    "src/{pkg}/pipeline",
    "src/{pkg}/utils",
    "src/{pkg}/exception",
    "src/{pkg}/logging",
    "config",
    "data/raw",
    "data/interim",
    "data/processed",
    "artifacts",
    "deployment/model",
    "logs",
    "mlruns",
    ".github/workflows",
    "tests/unit",
    "tests/integration",
    "notebooks",
    "feast_example",
    "feature_repo",
    "k8s",
    "helm/network-security-api/templates",
    "monitoring",
    "deployment",
    "terraform/modules/project_info",
    "terraform/modules/deployment",
    "terraform/modules/storage",
    "terraform/modules/container_registry",
    "terraform/modules/iam",
]


PYTHON_FILES = [
    "src/{pkg}/__init__.py",
    "src/{pkg}/app.py",
    "src/{pkg}/constants/__init__.py",
    "src/{pkg}/constants/training_pipeline.py",
    "src/{pkg}/entity/__init__.py",
    "src/{pkg}/entity/config_entity.py",
    "src/{pkg}/entity/artifact_entity.py",
    "src/{pkg}/components/__init__.py",
    "src/{pkg}/components/data_ingestion.py",
    "src/{pkg}/components/data_validation.py",
    "src/{pkg}/components/data_transformation.py",
    "src/{pkg}/components/feast_data_prepearation.py",
    "src/{pkg}/components/model_trainer.py",
    "src/{pkg}/components/model_evaluation.py",
    "src/{pkg}/components/model_registry.py",
    "src/{pkg}/components/model_pusher.py",
    "src/{pkg}/components/prediction.py",
    "src/{pkg}/components/schema.py",
    "src/{pkg}/pipeline/__init__.py",
    "src/{pkg}/pipeline/training_pipeline.py",
    "src/{pkg}/pipeline/prediction_pipeline.py",
    "src/{pkg}/pipeline/prefect_pipeline.py",
    "src/{pkg}/utils/__init__.py",
    "src/{pkg}/utils/main_utils.py",
    "src/{pkg}/exception/__init__.py",
    "src/{pkg}/exception/exception.py",
    "src/{pkg}/logging/__init__.py",
    "src/{pkg}/logging/logger.py",
    "tests/__init__.py",
    "tests/unit/test_data_transformation.py",
    "tests/unit/test_data_validation.py",
    "tests/unit/test_model_registry.py",
    "tests/unit/test_model_trainer.py",
    "tests/integration/test_data_ingestion.py",
    "feature_repo/__init__.py",
    "feature_repo/entities.py",
    "feature_repo/features.py",
    "feature_repo/sources.py",
    "feature_repo/feature_services.py",
    "test_postgres.py",
    "test_prediction.py",
    "test_prediction_pipeline.py",
]


def touch_files(root: Path, paths: list[str]) -> None:
    for relative in paths:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)


def create_project_structure(project_name: str) -> None:
    root = Path(project_name)
    package_name = root.name.lower().replace("-", "_").replace(" ", "_")

    for relative in DIRECTORIES:
        (root / relative.format(pkg=package_name)).mkdir(parents=True, exist_ok=True)

    touch_files(root, [p.format(pkg=package_name) for p in PYTHON_FILES])
    touch_files(root, STRUCTURAL_FILES)

    for relative, content in TEXT_TEMPLATES.items():
        path = root / relative.format(pkg=package_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        rendered = content.replace("{project_name}", project_name).replace("{package_name}", package_name)
        path.write_text(rendered, encoding="utf-8")

    for relative in [
        "data/raw/.gitkeep",
        "data/interim/.gitkeep",
        "data/processed/.gitkeep",
        "artifacts/.gitkeep",
        "logs/.gitkeep",
        "mlruns/.gitkeep",
        "notebooks/.gitkeep",
        "deployment/.gitkeep",
    ]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)

    print("=" * 64)
    print(f"Project '{project_name}' created successfully!")
    print("=" * 64)
    print(f"Package name : {package_name}")
    print(f"Project path : {root.resolve()}")
    print("\nNext steps:")
    print(f"  cd {project_name}")
    print("  python -m venv .venv")
    print("  .venv\\Scripts\\activate")
    print("  pip install -r requirements.txt")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python setup_project.py <project_name>")
        print("Example: python setup_project.py networksecurity")
        sys.exit(1)

    create_project_structure(sys.argv[1])
