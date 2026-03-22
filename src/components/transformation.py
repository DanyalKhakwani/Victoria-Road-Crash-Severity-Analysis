import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pickle
from src.logger import logger
from src.exception import CustomException

class DataTransformation:
    def __init__(self, input_path: str, output_dir: str):
        self.input_path = input_path
        self.output_dir = output_dir

    def keep_top_n(self, df, column, n):
        top_n = df[column].value_counts().nlargest(n).index
        df[column] = df[column].where(df[column].isin(top_n), other='OTHER')
        return df

    def run(self):
        try:
            logger.info("Loading raw data")
            df = pd.read_csv(self.input_path)

            # ── Fix dtypes ────────────────────────────────────────────
            logger.info("Fixing dtypes")
            df['severity']           = df['severity'].astype(int)
            df['max_vehicle_damage'] = pd.to_numeric(df['max_vehicle_damage'], errors='coerce')

            # ── Drop columns ──────────────────────────────────────────
            logger.info("Dropping unnecessary columns")
            df.drop(columns=['lga_name', 'accident_no'], inplace=True)

            # ── Remove day_of_week == 0 ───────────────────────────────
            logger.info("Removing day_of_week == 0")
            df = df[df['day_of_week'] != '0']

            # ── Target: binary severity ───────────────────────────────
            logger.info("Encoding target variable")
            df['severity'] = df['severity'].apply(lambda x: 1 if x == 1 else 0)

            # ── Date/time extraction ──────────────────────────────────
            logger.info("Extracting date and time features")
            df['year']  = pd.to_datetime(df['accident_date']).dt.year
            df['month'] = pd.to_datetime(df['accident_date']).dt.month
            df['hour']  = pd.to_datetime(df['accident_time'], format='%H:%M:%S.%f').dt.hour
            df.drop(columns=['accident_date', 'accident_time'], inplace=True)

            # ── Speed zone: replace garbage codes ─────────────────────
            logger.info("Cleaning speed_zone")
            df['speed_zone'] = df['speed_zone'].replace([999, 888, 777], np.nan)
            median_speed     = df['speed_zone'].median()
            df['speed_zone'] = df['speed_zone'].fillna(median_speed)

            # ── police_attended: replace 9 with mode ──────────────────
            logger.info("Cleaning police_attended")
            mode_police          = df['police_attended'].mode()[0]
            df['police_attended'] = df['police_attended'].replace(9, mode_police)

            # ── Nulls: fill categorical with UNKNOWN ──────────────────
            logger.info("Handling categorical nulls")
            for col in ['road_type', 'deg_urban_name', 'node_type']:
                df[col] = df[col].fillna('UNKNOWN')

            # ── Nulls: fill max_vehicle_damage with median ────────────
            df['max_vehicle_damage'] = df['max_vehicle_damage'].fillna(
                df['max_vehicle_damage'].median()
            )

            # ── Top N encoding ────────────────────────────────────────
            logger.info("Applying top N category grouping")
            df = self.keep_top_n(df, 'road_type', n=10)
            df = self.keep_top_n(df, 'dca_code',  n=25)

            # ── Convert all object columns to string ──────────────────
            logger.info("Converting categorical columns to string")
            for col in df.select_dtypes(include='object').columns:
                df[col] = df[col].astype(str)

            # ── Define features and target ────────────────────────────
            target = 'severity'
            X = df.drop(columns=[target])
            y = df[target]

            categorical_cols = X.select_dtypes(include='object').columns.tolist()
            numerical_cols   = X.select_dtypes(exclude='object').columns.tolist()

            logger.info(f"Categorical columns: {categorical_cols}")
            logger.info(f"Numerical columns: {numerical_cols}")

            # ── Preprocessor pipeline ─────────────────────────────────
            logger.info("Building preprocessor")
            numerical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler',  StandardScaler())
            ])

            categorical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])

            preprocessor = ColumnTransformer([
                ('num', numerical_pipeline, numerical_cols),
                ('cat', categorical_pipeline, categorical_cols)
            ])

            # ── Train/test split ──────────────────────────────────────
            logger.info("Splitting into train and test sets")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )

            logger.info(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

            # ── Fit and transform ─────────────────────────────────────
            logger.info("Fitting and transforming data")
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed  = preprocessor.transform(X_test)

            # ── Save train and test CSVs ──────────────────────────────
            os.makedirs(self.output_dir, exist_ok=True)

            train_df = pd.DataFrame(X_train_transformed)
            test_df  = pd.DataFrame(X_test_transformed)

            train_df['severity'] = y_train.values
            test_df['severity']  = y_test.values

            train_path = os.path.join(self.output_dir, 'train.csv')
            test_path  = os.path.join(self.output_dir, 'test.csv')

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path,   index=False)

            # ── Save preprocessor ─────────────────────────────────────
            preprocessor_path = os.path.join(self.output_dir, 'preprocessor.pkl')
            with open(preprocessor_path, 'wb') as f:
                pickle.dump(preprocessor, f)

            logger.info(f"Train saved  → {train_path}")
            logger.info(f"Test saved   → {test_path}")
            logger.info(f"Preprocessor → {preprocessor_path}")

            return train_path, test_path, preprocessor_path

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    transformation = DataTransformation(
        input_path="artifacts/raw_data.csv",
        output_dir="artifacts"
    )
    transformation.run()