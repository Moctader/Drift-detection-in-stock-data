from typing import Text

#DATABASE_URI = "postgresql://your_user:your_password@localhost:5432/your_database"
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base, DataDriftPredictionTable, DataQualityTable, ModelPerformanceTable, TargetDriftTable

# Database connection parameters
DATABASE_URI = "postgresql://your_user:your_password@localhost:5432/your_database"

# Create a database engine
engine = create_engine(DATABASE_URI)

# Create all tables in the database
Base.metadata.create_all(engine)

print("Tables created successfully.")