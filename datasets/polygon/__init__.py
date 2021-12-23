# -*- coding: utf-8 -*-

import datetime
from dotenv import load_dotenv
import os
from polygon import RESTClient

load_dotenv()
KEY = os.environ.get('API_KEY')


def ts_to_datetime(ts) -> str:
    return datetime.datetime.fromtimestamp(ts / 1000.0).strftime('%Y-%m-%d %H:%M')



with RESTClient(KEY) as client:
    resp = client.stocks_equities_daily_open_close("AAPL", "2021-06-11")
    print(f"On: {resp.from_} Apple opened at {resp.open} and closed at {resp.close}")
