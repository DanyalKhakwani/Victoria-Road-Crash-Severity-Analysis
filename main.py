from src.components.ingest import DataIngestion

if __name__ == "__main__":
    ingestion = DataIngestion(
        db_path="eda/vic_crash_data.db",
        output_path="artifacts/raw_data.csv"
    )
    ingestion.run()