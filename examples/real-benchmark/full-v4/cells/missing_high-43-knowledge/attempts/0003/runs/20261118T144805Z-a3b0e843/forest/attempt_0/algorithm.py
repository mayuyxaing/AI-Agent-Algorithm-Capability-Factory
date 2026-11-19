import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

def build_model(seed=42):
    """
    构建并返回一个未训练的随机森林分类器 Pipeline。
    
    参数:
    seed (int): 随机种子，确保结果可复现。
    
    返回:
    sklearn.pipeline.Pipeline: 包含预处理和分类器的 Pipeline 对象。
    """
    # 定义预处理步骤
    numeric_features = ['tenure', 'monthly_charges', 'support_calls']
    categorical_features = ['contract', 'payment_method']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # 组装 Pipeline
    preprocessor = Pipeline(steps=[
        ('num', numeric_transformer),
        ('cat', categorical_transformer)
    ])

    # 构建随机森林分类器，设置 class_weight 以处理类别不平衡
    classifier = RandomForestClassifier(
        n_estimators=100, 
        max_depth=7, 
        min_samples_leaf=5, 
        class_weight='balanced',
        random_state=seed,
        n_jobs=1
    )

    # 组装最终 Pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    return model

def predict(model, frame):
    """
    使用给定的模型对输入数据帧进行预测，返回正类的概率。
    
    参数:
    model (sklearn.pipeline.Pipeline): 训练好的模型。
    frame (pd.DataFrame): 待预测的数据帧。
    
    返回:
    np.ndarray: 正类的概率数组，长度与输入数据帧的行数一致。
    
    异常:
    ValueError: 当输入数据帧缺少必需的特征列时抛出。
    """
    # 检查必需的特征列是否存在
    required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']
    for feature in required_features:
        if feature not in frame.columns:
            raise ValueError(f"Missing required field: {feature}")

    # 调用模型的 predict_proba 方法，并提取正类的概率
    probabilities = model.predict_proba(frame)
    
    # 预测结果是 [prob_class_0, prob_class_1] 的形式，我们只需要 prob_class_1
    class_1_probabilities = probabilities[:, 1]
    
    # 确保概率是有限的数值（在极端情况下，如模型未拟合，可能会有 np.nan 或 np.inf）
    if not np.all(np.isfinite(class_1_probabilities)):
        raise ValueError("Predicted probabilities are not finite.")
    
    return class_1_probabilities