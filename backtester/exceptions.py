# -*- coding: utf-8 -*-


class TradingError(Exception):
    """Generic trading error."""
    pass

class NoMoneyError(TradingError):
    """Trading error when there is no available funds for a transaction."""
    pass


class DataError(Exception):
    pass

class NotEnoughDataError(DataError):
    """If you don't have enough data to perform a calculation."""
    pass

