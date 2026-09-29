import pandas as pd
from sqlalchemy import create_engine

df = pd.read_csv("data//raw//Customer-Churn.csv")

engine = create_engine("postgresql+psycopg2://postgres:Sauske%400167@localhost:5432/customer_churn")

df.to_sql("customer_churn", engine, if_exists='replace', index=False)

print(f"Inserted {len(df)} records sucessfully")