import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


def build_model(seed=42):
    """
    构建一个用于客户流失预测的Pipeline。
    
    参数:
    seed (int): 随机种子，用于模型的可复现性。
    
    返回:
    sklearn.pipeline.Pipeline: 一个未拟合的Pipeline，包含预处理步骤和逻辑回归分类器。
    """
    numeric_features = ['tenure', 'monthly_charges', 'support_calls']
    categorical_features = ['contract', 'payment_method']
    
    # 数值特征处理：中位数填补 + 标准化
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # 类别特征处理：众数填补 + 独热编码 (允许未知类别)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # 组合所有特征转换器
    preprocessor = Pipeline(steps=[
        ('num', numeric_transformer),
        ('cat', categorical_transformer)
    ])
    
    # 构建最终Pipeline，分类器为逻辑回归
    # 设置 class_weight='balanced' 以处理类别不平衡
    # 设置 solver='lbfgs' 以支持 class_weight
    clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight='balanced',
        solver='lbfgs',
        random_state=seed
    )
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    
    return model


def predict(model, frame):
    """
    对输入数据帧进行预测，返回正类概率。
    
    参数:
    model (sklearn.pipeline.Pipeline): 已训练的模型。
    frame (pd.DataFrame): 输入数据，必须包含所有必需特征。
    
    返回:
    np.ndarray: 正类概率数组，与输入行顺序一致。
    
    异常:
    ValueError: 如果输入数据缺少必需的特征列。
    """
    required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']
    missing_features = [f for f in required_features if f not in frame.columns]
    if missing_features:
        raise ValueError(f"缺少必需的特征列: {missing_features}")
    
    # predict_proba 返回形状为 (n_samples, n_classes) 的数组，取正类概率
    probabilities = model.predict_proba(frame)
    return probabilities[:, 1]
