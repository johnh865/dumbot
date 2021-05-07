# -*- coding: utf-8 -*-
import numpy as np
from definitions import TRADE_DATES_PATH

trade_dates = np.genfromtxt(TRADE_DATES_PATH, dtype=np.datetime64)