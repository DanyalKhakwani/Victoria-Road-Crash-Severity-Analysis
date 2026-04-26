import os
import sys
import json
import pickle
import pandas as pd
from datetime import datetime

from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    recall_score, roc_auc_score, precision_score,
    f1_score, accuracy_score, classification_report, confusion_matrix
)

from src.logger import logger
from src.exception import CustomException
from src.utils import load_config


class ModelTrainerSMOTE:
    def __init__(self, train_path: str, test_path: str, model_output_dir: str):
        self.train_path       = train_path
        self.test_path        = test_path
        self.model_output_dir = model_output_dir
        self.config           = load_config()['experiment']

    def save_experiment(self, model_name: str, metrics: dict):
        try:
            experiments_dir = "experiments"
            os.makedirs(experiments_dir, exist_ok=True)

            timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            filename  = f"{timestamp}_{model_name.lower().replace(' ', '_')}_smote.json"

            experiment = {
                "timestamp":  timestamp,
                "model_name": model_name,
                "sampling":   "SMOTE",
                "metrics":    metrics
            }

            path = os.path.join(experiments_dir, filename)
            with open(path, 'w') as f:
                json.dump(experiment, f, indent=4)

            logger.info(f"Experiment saved → {path}")

        except Exception as e:
            raise CustomException(e, sys)

    def get_model(self):
        model_name   = self.config['model']
        class_weight = self.config['class_weight']
        random_state = self.config['random_state']

        if model_name == "logistic_regression":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(
                class_weight=class_weight,
                max_iter=self.config['max_iter'],
                random_state=random_state
            ), "Logistic Regression"

        elif model_name == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                class_weight=class_weight,
                n_estimators=self.config['n_estimators'],
                random_state=random_state
            ), "Random Forest"

        elif model_name == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(
                n_estimators=self.config['n_estimators'],
                max_depth=self.config.get('max_depth', 3),          # add this
                learning_rate=self.config.get('learning_rate', 0.1),
                random_state=random_state
            ), "Gradient Boosting"

        else:
            raise ValueError(f"Unknown model: {model_name}")

    def run(self):
        try:
            # ── Load data ─────────────────────────────────────────────
            logger.info("Loading train and test data")
            train_df = pd.read_csv(self.train_path)
            test_df  = pd.read_csv(self.test_path)

            # ── Drop experiment columns ───────────────────────────────
            drop_cols = self.config.get('drop_columns', [])
            if drop_cols:
                logger.info(f"Dropping columns: {drop_cols}")
                train_df = train_df.drop(columns=drop_cols, errors='ignore')
                test_df  = test_df.drop(columns=drop_cols,  errors='ignore')

            X_train = train_df.drop(columns=['severity'])
            y_train = train_df['severity']
            X_test  = test_df.drop(columns=['severity'])
            y_test  = test_df['severity']

            # ── SMOTE oversampling ────────────────────────────────────
            smote_random_state = self.config.get('smote_random_state', self.config['random_state'])
            smote_k_neighbors  = self.config.get('smote_k_neighbors', 5)

            logger.info(
                f"Applying SMOTE (k_neighbors={smote_k_neighbors}, "
                f"random_state={smote_random_state})"
            )
            logger.info(f"Class distribution before SMOTE: {y_train.value_counts().to_dict()}")

            smote = SMOTE(
                k_neighbors=smote_k_neighbors,
                random_state=smote_random_state
            )
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

            import pandas as _pd
            y_resampled_series = _pd.Series(y_train_resampled)
            logger.info(f"Class distribution after SMOTE:  {y_resampled_series.value_counts().to_dict()}")

            # ── Train ─────────────────────────────────────────────────
            model, model_name = self.get_model()
            logger.info(f"Training {model_name} (SMOTE)")
            model.fit(X_train_resampled, y_train_resampled)

            # ── Evaluate ──────────────────────────────────────────────
            threshold   = self.config.get('threshold', 0.5)
            y_pred_prob = model.predict_proba(X_test)[:, 1]
            y_pred      = (y_pred_prob >= threshold).astype(int)

            recall    = recall_score(y_test, y_pred)
            roc_auc   = roc_auc_score(y_test, y_pred_prob)
            precision = precision_score(y_test, y_pred, pos_label=1)
            f1        = f1_score(y_test, y_pred, pos_label=1)
            accuracy  = accuracy_score(y_test, y_pred)

            print("\n── Classification Report (SMOTE) ──────────────────")
            print(classification_report(y_test, y_pred, target_names=['Non-Fatal', 'Fatal']))

            print("\n── Confusion Matrix ───────────────────────────────")
            cm = confusion_matrix(y_test, y_pred)
            print(f"                 Predicted Non-Fatal   Predicted Fatal")
            print(f"Actual Non-Fatal       {cm[0][0]:<10}        {cm[0][1]}")
            print(f"Actual Fatal           {cm[1][0]:<10}        {cm[1][1]}")

            print(f"\nRecall    (fatal): {recall:.4f}")
            print(f"Precision (fatal): {precision:.4f}")
            print(f"F1        (fatal): {f1:.4f}")
            print(f"ROC AUC:           {roc_auc:.4f}")
            print(f"Accuracy:          {accuracy:.4f}")

            # ── Save experiment ───────────────────────────────────────
            self.save_experiment(
                model_name=model_name,
                metrics={
                    "recall_fatal":         round(recall,    4),
                    "precision_fatal":      round(precision, 4),
                    "f1_fatal":             round(f1,        4),
                    "roc_auc":              round(roc_auc,   4),
                    "accuracy":             round(accuracy,  4),
                    "threshold":            threshold,
                    "dropped_columns":      drop_cols,
                    "smote_k_neighbors":    smote_k_neighbors,
                    "smote_random_state":   smote_random_state,
                    "parameters":           model.get_params()
                }
            )

            # ── Save model ────────────────────────────────────────────
            os.makedirs(self.model_output_dir, exist_ok=True)
            model_path = os.path.join(self.model_output_dir, 'model_smote.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

            logger.info(f"Model saved → {model_path}")
            return model_path

        except Exception as e:
            raise CustomException(e, sys)