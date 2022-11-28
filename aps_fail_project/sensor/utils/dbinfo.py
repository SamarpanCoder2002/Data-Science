from dataclasses import dataclass

@dataclass
class DBClass:
    DATABASE_NAME:str="aps"
    COLLECTION_NAME:str="sensor"