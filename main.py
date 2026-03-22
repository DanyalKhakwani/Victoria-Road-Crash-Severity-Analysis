from src.components.ingest import DataIngestion
from src.components.transformation import DataTransformation
from src.components.model_trainer import ModelTrainer



if __name__ == "__main__":
    # ingestion = DataIngestion(
    #     db_path="eda/vic_crash_data.db",
    #     output_path="artifacts/raw_data.csv"
    # )
    # ingestion.run()

    # transformation = DataTransformation(
    #     input_path="artifacts/raw_data.csv",
    #     output_dir="artifacts"
    # )
    # transformation.run()

    trainer = ModelTrainer(
        train_path="artifacts/train.csv",
        test_path="artifacts/test.csv",
        model_output_dir="artifacts"
    )
    trainer.run()