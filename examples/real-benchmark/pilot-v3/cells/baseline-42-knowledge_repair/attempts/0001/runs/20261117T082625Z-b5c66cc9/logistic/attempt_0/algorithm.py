import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin

def build_model(seed=42):
    """
    构建并返回一个未拟合的机器学习 Pipeline。

    参数:
        seed (int): 随机种子，用于逻辑回归和随机森林的随机状态设置。

    返回:
        sklearn.pipeline.Pipeline: 包含预处理和分类器的未拟合 Pipeline。
    """
    # 定义特征列
    numeric_features = ['tenure', 'monthly_charges', 'support_calls']
    categorical_features = ['contract', 'payment_method']
    required_features = numeric_features + categorical_features

    # 数值特征预处理：中位数填补，然后标准化
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 类别特征预处理：众数填补，然后独热编码（允许未知类别）
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # 组合预处理步骤
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 定义候选分类器：逻辑回归
    # 使用 class_weight='balanced' 处理类别不平衡
    # max_iter 设置为 1000 以确保收敛
    classifier = LogisticRegression(
        C=1.0, 
        max_iter=1000, 
        class_weight='balanced', 
        solver='lbfgs', 
        random_state=seed
    )

    # 创建 Pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    return model

def predict(model, frame):
    """
    使用给定的模型对输入数据帧进行预测。

    参数:
        model (sklearn.pipeline.Pipeline): 已拟合的模型。
        frame (pd.DataFrame): 包含特征的输入数据。

    返回:
        numpy.ndarray: 每行输入属于正类（churn）的预测概率，顺序与输入一致。

    抛出:
        ValueError: 如果输入缺少任何必需的特征。
    """
    # 检查必需特征是否存在
    required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']
    missing_features = [f for f in required_features if f not in frame.columns]
    if missing_features:
        raise ValueError(f"Missing required fields: {missing_features}")

    # 确保输入是 DataFrame
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)

    # 预测概率（正类的概率）
    proba = model.predict_proba(frame)

    # 返回正类概率列
    return proba[:, 1]