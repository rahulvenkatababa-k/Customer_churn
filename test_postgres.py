import pandas as pd
from sqlalchemy import create_engine

from src.customer_churn.configuration.configuration import config

df = pd.read_csv("data//raw//Customer-Churn.csv")

engine = create_engine(config.postgres_sql_uri)

df.to_sql("customer_churn", engine, if_exists='replace', index=False)

print(f"Inserted {len(df)} records sucessfully")