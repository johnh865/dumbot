# -*- coding: utf-8 -*-
from os.path import dirname, join

PACKAGE_PATH = dirname(__file__)
PROJECT_PATH = dirname(PACKAGE_PATH)
ROI_URL = 'sqlite:///' + join(PACKAGE_PATH, 'roi.db.sqlite3')
ROLLING_URL = 'sqlite:///' + join(PACKAGE_PATH, 'rolling.db.sqlite3')

ROI_DIR = join(PACKAGE_PATH, 'roi.parquet')
ROLLING_DIR = join(PACKAGE_PATH, 'rolling.parquet')




# FILE_ROI = join(PACKAGE_PATH, 'monthly_ROI.csv')
# FILE_STATS = join(PACKAGE_PATH, 'monthly_stats.csv')
# FILE_ROLLING_STATS = join(PACKAGE_PATH, 'rolling_12_month_stats.csv')
