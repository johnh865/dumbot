# -*- coding: utf-8 -*-
from datasets.yahoo.definitions import CONNECTION_PATH
from backtester.utils import SQLClient


client = SQLClient(CONNECTION_PATH)

engine2 = client.to_memory()