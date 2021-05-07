# -*- coding: utf-8 -*-


from os.path import dirname, join

PACKAGE_PATH = dirname(__file__)
PROJECT_PATH = dirname(PACKAGE_PATH)
TRADE_DATES_PATH = join(PACKAGE_PATH, 'trade_dates.txt')