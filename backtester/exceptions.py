# -*- coding: utf-8 -*-


class TradingError(Exception):
    pass

class NoMoneyError(TradingError):
    pass
