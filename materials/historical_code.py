"""Historical code used as extraction input; it is not imported by the workflow."""
from sklearn.preprocessing import OneHotEncoder


def churn_category_encoder():
    """客户流失预测：合同类别编码应允许未知类别，避免上线预测失败。"""
    return OneHotEncoder(handle_unknown="ignore")
