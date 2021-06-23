# -*- coding: utf-8 -*-
from os.path import dirname, join

PACKAGE_PATH = dirname(__file__)

FILE_ROI = join(PACKAGE_PATH, 'monthly_ROI.csv')
FILE_STATS = join(PACKAGE_PATH, 'monthly_stats.csv')
FILE_ROLLING_STATS = join(PACKAGE_PATH, 'rolling_12_month_stats.csv')
