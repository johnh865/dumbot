# -*- coding: utf-8 -*-

from dotenv import load_dotenv
import os
from polygon import RESTClient

load_dotenv()
KEY = os.environ.get('API_KEY')

