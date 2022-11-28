from sensor.logger import logging
from sensor.exception import SensorException
from sensor.utils import get_collection_as_dataframe
from sensor.utils.dbinfo import DBClass
import os
import sys


if __name__ == '__main__':
    try:
        get_collection_as_dataframe(DBClass.DATABASE_NAME, DBClass.COLLECTION_NAME)
    except Exception as e:
        print(e)
