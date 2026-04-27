import os
from dataclasses import dataclass

@dataclass
class DatabaseConfig:
    db_path: str = os.getenv('DB_PATH', 'teammate_finder.db')
    
config = DatabaseConfig()