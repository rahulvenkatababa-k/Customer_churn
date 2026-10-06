import os
import sys
import optuna
from src.customer_churn.logging.logger import logger
from src.customer_churn.exception.exception import CustomerChurnException
from src.customer_churn.utils.utils import read_csv, save_object, save_json
from src.customer_churn.entity.artifact_entity import (
    FeatureEngineeringArtifact,
    ModelTrainerArtifact,
)
from src.customer_churn.entity.config_entity import ModelTrainerConfig

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score, cross_validate
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

from sklearn.base import clone

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler



class ModelTrainer:
    def __init__(
        self,
        feature_engineering_artifact: FeatureEngineeringArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.feature_engineering_artifact = feature_engineering_artifact
        self.model_trainer_config = model_trainer_config

    NUMERICAL_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]
    BINARY_COLUMNS = ["SeniorCitizen"]

    def _create_preprocessor(self, X_train):
        numerical_columns = self.NUMERICAL_COLUMNS
        binary_columns = self.BINARY_COLUMNS

        categorical_columns = [
            column for column in X_train.columns
            if column not in numerical_columns + binary_columns
        ]

        return ColumnTransformer(
            transformers=[
                ("numerical", StandardScaler(), numerical_columns),
                ("binary", "passthrough", binary_columns),
                (
                    "categorical",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                    categorical_columns,
                ),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

    def _load_data(self):
        try:
            X_train = read_csv(
                self.feature_engineering_artifact.X_train_path
            )

            y_train = read_csv(
                self.feature_engineering_artifact.y_train_path
            )

            y_train = y_train.squeeze()

            return X_train, y_train

        except Exception as e:
            logger.error("%s", e)
            raise CustomerChurnException(
                e, sys, "Model Trainer", "load data"
            ) from e

    def _baseline_model(self, X_train, y_train):
        try:
            model = Pipeline([
                ("preprocessor", self._create_preprocessor(X_train)),
                (
                    "model",
                    LogisticRegression(
                        random_state=self.model_trainer_config.random_state
                    ),
                ),
            ])

            model.fit(X_train, y_train)
            return model

        except Exception as e:
            logger.error("%s", e)
            raise CustomerChurnException(
                e, sys, "Model Trainer", "baseline model"
            ) from e

    def _calculate_sample_weights(self, y_train):
        try:
            sample_weights = compute_sample_weight(
                class_weight="balanced",
                y=y_train
            )

            return sample_weights

        except Exception as e:
            logger.error("%s", e)
            raise CustomerChurnException(e,sys,"Model Trainer","calculate sample weights") from e

    def _candidate_models(self, X_train, y_train):
        try:
            models = {}

            # AdaBoost
            ada_boost = Pipeline([
                ("preprocessor", self._create_preprocessor(X_train)),
                ("model", AdaBoostClassifier(random_state=self.model_trainer_config.random_state),),
            ])

            ada_boost.fit(X_train, y_train)

            models["adaboost"] = ada_boost

            # adaboost with balanced sample weights

            sample_weights = self._calculate_sample_weights(y_train)

            weighted_ada_boost = Pipeline([
                ("preprocessor", self._create_preprocessor(X_train)),
                (
                    "model",
                    AdaBoostClassifier(
                        random_state=self.model_trainer_config.random_state
                    ),
                ),
            ])

            weighted_ada_boost.fit(
                X_train,
                y_train,
                model__sample_weight=sample_weights,
            )

            models["adaboost_weighted"] = weighted_ada_boost

            # XGBoost
            xgboost = Pipeline([
                ("preprocessor", self._create_preprocessor(X_train)),
                (
                    "model",XGBClassifier(
                                    random_state=self.model_trainer_config.random_state,
                                    eval_metric="logloss",
                                )
                )
            ])
            
            
            
            

            xgboost.fit(X_train, y_train)

            models["xgboost"] = xgboost

            # CatBoost
            catboost = Pipeline([
                ("preprocessor", self._create_preprocessor(X_train)),
                ("model",CatBoostClassifier(
                                random_seed=self.model_trainer_config.random_state,
                                verbose=False,
                            ))
            ])
            
            

            catboost.fit(X_train, y_train)

            models["catboost"] = catboost

            return models

        except Exception as e:
            logger.error("%s", e)
            raise CustomerChurnException(
                e, sys, "Model Trainer", "candidate models"
            ) from e

    def _run_hyperparameter_experiments(self, X_train, y_train):
        try:
            cv = StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=self.model_trainer_config.random_state
            )

            experiment_results = {}

            # ============================================================
            # AdaBoost + SMOTE
            # ============================================================

            ada_smote_pipeline = Pipeline([
                ("preprocessor", self._create_preprocessor(X_train)),
                ("smote", SMOTE(
                    random_state=self.model_trainer_config.random_state
                )),
                ("model", AdaBoostClassifier(
                    random_state=self.model_trainer_config.random_state
                ))
            ])

            ada_smote_grid = {
                "model__n_estimators": [50, 100, 150],
                "model__learning_rate": [0.05, 0.1, 0.5, 1.0]
            }

            ada_smote_search = GridSearchCV(
                estimator=ada_smote_pipeline,
                param_grid=ada_smote_grid,
                scoring="f1",
                cv=cv,
                n_jobs=1,
                verbose=1
            )

            ada_smote_search.fit(X_train, y_train)

            experiment_results["adaboost_smote"] = {
                "best_score": ada_smote_search.best_score_,
                "best_params": ada_smote_search.best_params_,
                "best_estimator": ada_smote_search.best_estimator_
            }

            logger.info(
                "AdaBoost + SMOTE | CV f1: %.4f | Params: %s",
                ada_smote_search.best_score_,
                ada_smote_search.best_params_
            )

            # ============================================================
            # AdaBoost + SMOTEENN
            # ============================================================

            ada_smoteenn_pipeline = Pipeline([
                ("preprocessor", self._create_preprocessor(X_train)),
                ("smoteenn", SMOTEENN(
                    random_state=self.model_trainer_config.random_state
                )),
                ("model", AdaBoostClassifier(
                    random_state=self.model_trainer_config.random_state
                ))
            ])

            ada_smoteenn_grid = {
                "model__n_estimators": [50, 100, 150],
                "model__learning_rate": [0.05, 0.1, 0.5, 1.0]
            }

            ada_smoteenn_search = GridSearchCV(
                estimator=ada_smoteenn_pipeline,
                param_grid=ada_smoteenn_grid,
                scoring="f1",
                cv=cv,
                n_jobs=1,
                verbose=1
            )

            ada_smoteenn_search.fit(X_train, y_train)

            experiment_results["adaboost_smoteenn"] = {
                "best_score": ada_smoteenn_search.best_score_,
                "best_params": ada_smoteenn_search.best_params_,
                "best_estimator": ada_smoteenn_search.best_estimator_
            }

            logger.info(
                "AdaBoost + SMOTEENN | CV f1: %.4f | Params: %s",
                ada_smoteenn_search.best_score_,
                ada_smoteenn_search.best_params_
            )

            # ============================================================
            # XGBoost + scale_pos_weight
            # ============================================================

            negative_count = (y_train == 0).sum()
            positive_count = (y_train == 1).sum()

            scale_pos_weight = negative_count / positive_count

            xgb_model =Pipeline([ 

                ("preprocessor", self._create_preprocessor(X_train)),
                ("model", XGBClassifier(
                    random_state=self.model_trainer_config.random_state,
                    eval_metric="logloss"
            ))])

            xgb_grid = {
                    "model__n_estimators": [100, 200],
                    "model__max_depth": [3, 5, 7],
                    "model__learning_rate": [0.05, 0.1],
                    "model__scale_pos_weight": [1, scale_pos_weight],
                }

            xgb_search = GridSearchCV(
                estimator=xgb_model,
                param_grid=xgb_grid,
                scoring="f1",
                cv=cv,
                n_jobs=1,
                verbose=1
            )

            xgb_search.fit(X_train, y_train)

            experiment_results["xgboost_weighted"] = {
                "best_score": xgb_search.best_score_,
                "best_params": xgb_search.best_params_,
                "best_estimator": xgb_search.best_estimator_
            }

            logger.info(
                "XGBoost + class weighting | CV f1: %.4f | Params: %s",
                xgb_search.best_score_,
                xgb_search.best_params_
            )

            # ============================================================
            # XGBoost + SMOTE
            # ============================================================

            xgb_smote_pipeline = Pipeline([
                ("preprocessor", self._create_preprocessor(X_train)),
                ("smote", SMOTE(
                    random_state=self.model_trainer_config.random_state
                )),
                ("model", XGBClassifier(
                    random_state=self.model_trainer_config.random_state,
                    eval_metric="logloss"
                ))
            ])

            xgb_smote_grid = {
                "model__n_estimators": [100, 200],
                "model__max_depth": [3, 5],
                "model__learning_rate": [0.05, 0.1]
            }

            xgb_smote_search = GridSearchCV(
                estimator=xgb_smote_pipeline,
                param_grid=xgb_smote_grid,
                scoring="f1",
                cv=cv,
                n_jobs=1,
                verbose=1
            )

            xgb_smote_search.fit(X_train, y_train)

            experiment_results["xgboost_smote"] = {
                "best_score": xgb_smote_search.best_score_,
                "best_params": xgb_smote_search.best_params_,
                "best_estimator": xgb_smote_search.best_estimator_
            }

            logger.info(
                "XGBoost + SMOTE | CV f1: %.4f | Params: %s",
                xgb_smote_search.best_score_,
                xgb_smote_search.best_params_
            )

            # ============================================================
            # CatBoost + class weighting
            # ============================================================

            catboost_model = Pipeline([
                ("preprocessor", self._create_preprocessor(X_train)),
                ("model", CatBoostClassifier(
                                random_seed=self.model_trainer_config.random_state,
                                verbose=False
                            ) )
            ])

            catboost_grid = {
                "model__iterations": [100, 200],
                "model__depth": [4, 6, 8],
                "model__learning_rate": [0.05, 0.1],
                "model__class_weights": [
                    [1, 1],
                    [1, scale_pos_weight],
                ],
            }


            catboost_search = GridSearchCV(
                estimator=catboost_model,
                param_grid=catboost_grid,
                scoring="f1",
                cv=cv,
                n_jobs=1,
                verbose=1
            )

            catboost_search.fit(X_train, y_train)

            experiment_results["catboost_weighted"] = {
                "best_score": catboost_search.best_score_,
                "best_params": catboost_search.best_params_,
                "best_estimator": catboost_search.best_estimator_
            }

            logger.info(
                "CatBoost + class weighting | CV f1: %.4f | Params: %s",
                catboost_search.best_score_,
                catboost_search.best_params_
            )

            return experiment_results

        except Exception as e:
            logger.error("%s", e)
            raise CustomerChurnException(
                e,
                sys,
                "Model Trainer",
                "run hyperparameter experiments"
            ) from e

    def _run_optuna_experiments(self, X_train, y_train):
        try:
            cv = StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=self.model_trainer_config.random_state
            )

            optuna_results = {}

            # ============================================================
            # AdaBoost + SMOTEENN
            # ============================================================

            def smoteenn_objective(trial):

                model = AdaBoostClassifier(
                    n_estimators=trial.suggest_int(
                        "n_estimators", 50, 300
                    ),
                    learning_rate=trial.suggest_float(
                        "learning_rate", 0.01, 1.0, log=True
                    ),
                    random_state=self.model_trainer_config.random_state
                )

                pipeline = Pipeline([
                    ("preprocessor", self._create_preprocessor(X_train)),
                    (
                        "smoteenn",
                        SMOTEENN(
                            random_state=self.model_trainer_config.random_state
                        )
                    ),
                    ("model", model)
                ])

                scores = cross_val_score(
                    pipeline,
                    X_train,
                    y_train,
                    scoring="f1",
                    cv=cv,
                    n_jobs=1
                )

                return scores.mean()

            smoteenn_study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(
                    seed=self.model_trainer_config.random_state
                )
            )

            smoteenn_study.optimize(
                smoteenn_objective,
                n_trials=20,
                show_progress_bar=False
            )

            smoteenn_best_model = Pipeline([
                ("preprocessor", self._create_preprocessor(X_train)),
                (
                    "smoteenn",
                    SMOTEENN(
                        random_state=self.model_trainer_config.random_state
                    )
                ),
                (
                    "model",
                    AdaBoostClassifier(
                        n_estimators=smoteenn_study.best_params["n_estimators"],
                        learning_rate=smoteenn_study.best_params["learning_rate"],
                        random_state=self.model_trainer_config.random_state
                    )
                )
            ])

            smoteenn_best_model.fit(X_train, y_train)

            optuna_results["adaboost_smoteenn"] = {
                "best_score": float(smoteenn_study.best_value),
                "best_params": smoteenn_study.best_params,
                "best_estimator": smoteenn_best_model
            }

            logger.info(
                "Optuna AdaBoost + SMOTEENN | CV f1: %.4f | Params: %s",
                smoteenn_study.best_value,
                smoteenn_study.best_params
            )

            # ============================================================
            # AdaBoost + SMOTE
            # ============================================================

            def smote_objective(trial):

                model = AdaBoostClassifier(
                    n_estimators=trial.suggest_int(
                        "n_estimators", 50, 300
                    ),
                    learning_rate=trial.suggest_float(
                        "learning_rate", 0.01, 1.0, log=True
                    ),
                    random_state=self.model_trainer_config.random_state
                )

                pipeline = Pipeline([
                    ("preprocessor", self._create_preprocessor(X_train)),
                    (
                        "smote",
                        SMOTE(
                            random_state=self.model_trainer_config.random_state
                        )
                    ),
                    ("model", model)
                ])

                scores = cross_val_score(
                    pipeline,
                    X_train,
                    y_train,
                    scoring="f1",
                    cv=cv,
                    n_jobs=1
                )

                return scores.mean()

            smote_study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(
                    seed=self.model_trainer_config.random_state
                )
            )

            smote_study.optimize(
                smote_objective,
                n_trials=20,
                show_progress_bar=False
            )

            smote_best_model = Pipeline([
                ("preprocessor", self._create_preprocessor(X_train)),
                (
                    "smote",
                    SMOTE(
                        random_state=self.model_trainer_config.random_state
                    )
                ),
                (
                    "model",
                    AdaBoostClassifier(
                        n_estimators=smote_study.best_params["n_estimators"],
                        learning_rate=smote_study.best_params["learning_rate"],
                        random_state=self.model_trainer_config.random_state
                    )
                )
            ])

            smote_best_model.fit(X_train, y_train)

            optuna_results["adaboost_smote"] = {
                "best_score": float(smote_study.best_value),
                "best_params": smote_study.best_params,
                "best_estimator": smote_best_model
            }

            logger.info(
                "Optuna AdaBoost + SMOTE | CV f1: %.4f | Params: %s",
                smote_study.best_value,
                smote_study.best_params
            )

            return optuna_results

        except Exception as e:
            logger.error("%s", e)
            raise CustomerChurnException(
                e,
                sys,
                "Model Trainer",
                "run Optuna experiments"
            ) from e
        
    def _save_model(self, model, model_name, model_directory=None):
        try:
            if model_directory is None:
                model_directory = os.path.dirname(
                    self.model_trainer_config.model_path
                )

            model_path = os.path.join(
                model_directory,
                f"{model_name}.pkl"
            )

            save_object(model, model_path)

            return model_path

        except Exception as e:
            logger.error("%s", e)
            raise CustomerChurnException(
                e, sys, f"save {model_name}"
            ) from e

    def _save_experiment_results(self, experiment_results):
        try:
            experiment_models_directory = os.path.join(
                self.model_trainer_config.model_trainer_dir,
                "experiment_models"
            )

            experiment_results_path = os.path.join(
                self.model_trainer_config.model_trainer_dir,
                "experiment_results.json"
            )

            json_results = {}

            for experiment_name, result in experiment_results.items():

                model_path = self._save_model(
                model=result["best_estimator"],
                model_name=experiment_name,
                model_directory=experiment_models_directory
               )

                json_results[experiment_name] = {
                    "best_score": float(result["best_score"]),
                    "best_params": result["best_params"],
                    "model_path": model_path
                }

            save_json(
                obj=json_results,
                file_path=experiment_results_path,
                indent=4
            )

            return experiment_results_path

        except Exception as e:
            logger.error("%s", e)
            raise CustomerChurnException(
                e,
                sys,
                "Model Trainer",
                "save experiment results"
            ) from e


    def initiate_model_trainer(self):
        try:
            X_train, y_train = self._load_data()

            # Baseline model
            baseline_model = self._baseline_model(
                X_train=X_train,
                y_train=y_train,
            )

            baseline_model_path = self._save_model(
                model=baseline_model,
                model_name="baseline_model",
            )

            # Candidate models
            candidate_models = self._candidate_models(
                X_train=X_train,
                y_train=y_train,
            )

            # GridSearchCV experiments
            gridsearch_results = self._run_hyperparameter_experiments(
                X_train=X_train,
                y_train=y_train,
            )

            # Optuna experiments
            optuna_results = self._run_optuna_experiments(
                X_train=X_train,
                y_train=y_train,
            )

            # Keep GridSearchCV and Optuna results separate
            experiment_results = {}

            for name, result in gridsearch_results.items():
                if name in ("adaboost_smote", "adaboost_smoteenn"):
                    experiment_results[f"{name}_gridsearch"] = result
                else:
                    experiment_results[name] = result

            for name, result in optuna_results.items():
                experiment_results[f"{name}_optuna"] = result

            # Save experiment results
            experiment_results_path = self._save_experiment_results(
                experiment_results=experiment_results,
            )

            # Save candidate models
            candidate_model_paths = {}

            for model_name, model in candidate_models.items():
                model_path = self._save_model(
                    model=model,
                    model_name=model_name,
                )

                candidate_model_paths[model_name] = model_path

            # Create trainer artifact
            model_trainer_artifact = ModelTrainerArtifact(
                baseline_model_path=baseline_model_path,
                candidate_model_paths=candidate_model_paths,
                experiment_results_path=experiment_results_path,
            )

            return model_trainer_artifact

        except Exception as e:
            logger.error("%s", e)
            raise CustomerChurnException(
                e, sys, "Model Trainer", "initiate model trainer"
            ) from e