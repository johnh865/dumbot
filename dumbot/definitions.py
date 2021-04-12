# -*- coding: utf-8 -*-

"""Define constants here."""
from os.path import dirname, join

CONNECTION_NAME = 'sqlite:///db.sqlite3'
PACKAGE_PATH = dirname(__file__)
PROJECT_PATH = dirname(PACKAGE_PATH)
CONNECTION_PATH = 'sqlite:///' + join(PACKAGE_PATH, 'data', 'db.sqlite3')
