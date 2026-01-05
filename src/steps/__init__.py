import numpy as np

def compute_price_thresholds(prices):
    """
    Compute thresholds for price categorization.
    """
    low = np.percentile(prices, 33)
    high = np.percentile(prices, 66)
    return low, high

def assign_price_category(price, low, high):
    """
    Assign price category based on thresholds.
    """
    if price <= low:
        return "cheap"
    elif price <= high:
        return "mid"
    else:
        return "luxury"
