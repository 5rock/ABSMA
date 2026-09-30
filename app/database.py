from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import SQLAlchemyError
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///absma.db")

if DATABASE_URL.startswith("sqlite:///"):
    # If it's a relative sqlite path, make it absolute to the app directory
    db_file = DATABASE_URL.replace("sqlite:///", "")
    app_dir = os.path.dirname(os.path.abspath(__file__))
    DATABASE_URL = f"sqlite:///{os.path.join(app_dir, db_file)}"


if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL, echo=False)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def init_db():
    try:
        import models
        Base.metadata.create_all(bind=engine)
        print("[DB] Database initialized successfully.")
    except Exception as e:
        print(f"[DB ERROR] Could not initialize database: {e}")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
