# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from backtester.utils import datetime_to_np
import datetime


def test():
    """Test usage of datetime to numpy converter."""
    
    date1 = datetime.datetime(2012, 1, 1)
    date2 = np.datetime64(date1)
    date3 = pd.Timestamp(date1)
    
    dates1 = [date2, date2 + 521000]
    dates2 = np.array(dates1)
    
    out1 = datetime_to_np(date1)
    out2 = datetime_to_np(date2)
    out3 = datetime_to_np(date3)
    
    outs1 = datetime_to_np(dates1)
    outs2 = datetime_to_np(dates2)
    
    assert out1 == date2
    assert out2 == date2
    assert out3 == date2
    
    assert np.all(outs1 == dates2)
    assert np.all(outs2 == dates2)
    
    return
    
    
if __name__ == '__main__':
    test()