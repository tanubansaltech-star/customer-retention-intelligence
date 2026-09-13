import sqlite3
import pandas as pd


def load_customers_from_database():

    database_path = "data/customer_retention.db"

    connection = sqlite3.connect(database_path)

    df = pd.read_sql(
        "SELECT * FROM customers",
        connection
    )

    connection.close()

    return df