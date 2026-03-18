import os
import sys
import sqlite3
import pandas as pd
from src.logger import logger
from src.exception import CustomException

class DataIngestion:
    def __init__(self, db_path: str, output_path: str):
        self.db_path = db_path
        self.output_path = output_path

    def run(self) -> str:
        try:
            logger.info("Connecting to database")
            conn = sqlite3.connect(self.db_path)

            query = """
                SELECT 
                -- Accidents core
                a.accident_no,
                a.accident_type,
                a.day_of_week,
                a.dca_code,
                a.no_of_vehicles,
                a.police_attended,
                a.road_geometry,
                a.light_condition,
                a.speed_zone,
                a.severity,
                
                -- Date/time (extract in pandas)
                a.accident_date,
                a.accident_time,

                -- Node info
                n.node_type,
                n.deg_urban_name,
                n.lga_name,

                -- Road info
                an.road_type,

                -- Vehicle aggregates
                COUNT(DISTINCT v.vehicle_id)   as total_vehicles,
                MAX(v.level_of_damage)         as max_vehicle_damage,

                -- Person aggregates
                COUNT(DISTINCT p.person_id)    as total_persons

            FROM accidents a
            LEFT JOIN accident_node an  ON a.accident_no = an.accident_no
            LEFT JOIN road_node n       ON an.node_id    = n.node_id
            LEFT JOIN vehicle_info v    ON a.accident_no = v.accident_no
            LEFT JOIN person_info p     ON a.accident_no = p.accident_no

            GROUP BY a.accident_no
            """

            logger.info("Running ingestion query")
            df = pd.read_sql_query(query, conn)
            conn.close()

            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            df.to_csv(self.output_path, index=False)

            logger.info(f"Data ingested: {df.shape[0]} rows, {df.shape[1]} columns → {self.output_path}")
            return self.output_path

        except Exception as e:
            raise CustomException(e, sys)


