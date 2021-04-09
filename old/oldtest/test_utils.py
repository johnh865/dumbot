# -*- coding: utf-8 -*-

import datetime
import numpy as np
from dumbot.utils import get_trading_days, get_rolling_average, read_dataframe
from dumbot.utils import floor_to_days
import matplotlib.pyplot as plt
from dumbot import definitions

def test_get_trading_days():
    date = datetime.date(1995, 4, 2)
    t_days = get_trading_days(date)
    assert np.all(t_days >= np.datetime64(date))
    assert len(t_days) > 1
    
    return


def test_get_trading_days2():
    date1 = datetime.date(1997, 4, 2)
    date2 = datetime.date(2006, 1, 1)
    t_days = get_trading_days(date1, date2)
    assert np.all(t_days >= np.datetime64(date1))
    assert np.all(t_days <= np.datetime64(date2))

    

def test_rolling_average():
    """Just see if this runs."""
    df = read_dataframe('MSFT')
    df2 = get_rolling_average(df, 21)
    
    plt.subplot(2,1,1)
    plt.plot(df.index, df[definitions.DF_ADJ_CLOSE])
    plt.plot(df2.index, df2[definitions.DF_SMOOTH_CLOSE])
    plt.grid()
    plt.subplot(2,1,2)
    plt.plot(df2.index, df2[definitions.DF_SMOOTH_CHANGE])
    plt.axhline(0, color='k')
    plt.grid()
    return



    

if __name__ == '__main__':
    test_get_trading_days()
    test_get_trading_days2()
    test_rolling_average()
    test_floor_to_days()