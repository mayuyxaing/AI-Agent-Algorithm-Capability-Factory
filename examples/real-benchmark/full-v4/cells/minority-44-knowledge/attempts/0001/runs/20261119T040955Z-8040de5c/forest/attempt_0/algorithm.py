import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


def build_model(seed=42):
    """
    构建并返回一个未拟合的 sklearn Pipeline，用于客户流失预测。
    
    参数:
    seed (int): 随机种子。
    
    返回:
    sklearn.pipeline.Pipeline: 未拟合的模型管道。
    """
    numeric_features = ['tenure', 'monthly_charges', 'support_calls']
    categorical_features = ['contract', 'payment_method']

    # 预处理器：数值特征用中位数填补，然后标准化；类别特征用众数填补并做独热编码，忽略未知类别
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 分类器：随机森林，设置随机种子，n_jobs=1
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=7,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=seed,
        n_jobs=1
    )

    # 组装 Pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    return model


def predict(model, frame):
    """
    使用给定的模型对输入的 DataFrame 进行预测，返回正类（流失）的概率。
    
    参数:
    model (sklearn.pipeline.Pipeline): 拟合后的模型。
    frame (pd.DataFrame): 输入数据，必须包含 'tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method' 列。
    
    返回:
    np.ndarray: 1D 数组，包含每个样本的正类概率，顺序与输入一致。
    
    异常:
    ValueError: 如果输入数据缺少必需的特征列。
    """
    required_features = ['tenure', 'monthly_charges', 'support_calls', 'contract', 'payment_method']
    missing_fields = [f for f in required_features if f not in frame.columns]
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")

    # predict_proba 返回 shape (n_samples, n_classes)，我们取第 1 类（索引 1）的概率
    probabilities = model.predict_proba(frame)
    return probabilities[:, 1]


# 模块导入时不应进行训练或评估
if __name__ == '__main__':
    pass