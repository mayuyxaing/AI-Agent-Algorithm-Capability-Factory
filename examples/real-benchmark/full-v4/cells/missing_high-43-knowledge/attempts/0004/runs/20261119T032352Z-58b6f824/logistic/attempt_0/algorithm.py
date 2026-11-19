import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

def build_model(seed=42):
    """
    构建一个用于客户流失预测的 sklearn Pipeline。
    
    参数:
        seed (int): 随机种子，用于 LogisticRegression。
    
    返回:
        sklearn.pipeline.Pipeline: 未拟合的 Pipeline。
    """
    numeric_features = ['tenure', 'monthly_charges', 'support_calls']
    categorical_features = ['contract', 'payment_method']
    
    # 数值特征处理：中位数填补，标准化
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 类别特征处理：众数填补，独热编码（允许未知类别）
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # 组合预处理步骤
    preprocessor = Pipeline(steps=[
        ('num', numeric_transformer),
        ('cat', categorical_transformer)
    ])

    # 分类器：逻辑回归，使用 class_weight='balanced' 处理类别不平衡
    clf = LogisticRegression(
        C=1.0, 
        max_iter=1000, 
        class_weight='balanced', 
        solver='lbfgs',
        random_state=seed
    )

    # 完整的 Pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])

    return model

def predict(model, frame):
    """
    使用已训练模型对输入数据进行预测。
    
    参数:
        model (sklearn.pipeline.Pipeline): 已训练的模型。
        frame (pd.DataFrame): 包含特征数据的 DataFrame。
        
    返回:
        np.ndarray: 正类（流失）的预测概率，1D 数组。
    
    异常:
        ValueError: 当输入缺少必需的特征时抛出。
    """
    # 检查必需特征是否存在
    required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']
    missing_fields = [col for col in required_features if col not in frame.columns]
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")

    # 获取预测概率
    # Pipeline 的 predict_proba 方法会返回所有类别的概率，我们取正类（索引 1）的概率
    probabilities = model.predict_proba(frame)
    return probabilities[:, 1]