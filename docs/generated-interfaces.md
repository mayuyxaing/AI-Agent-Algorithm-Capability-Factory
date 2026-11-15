# 自动生成的任务与算法接口目录

来源为实际注册表；非 HTTP API 文档。

```json
{
  "tasks": {
    "churn_classification": {
      "target": "churn",
      "features": [
        "tenure",
        "monthly_charges",
        "support_calls",
        "contract",
        "payment_method"
      ],
      "interface": {
        "build_model": "build_model(seed=42) returns an unfitted sklearn.pipeline.Pipeline",
        "predict": "predict(model, frame) returns finite 1D positive-class probabilities in input row order, values [0,1]",
        "missing_fields": "raise ValueError when a required feature is absent",
        "features": [
          "tenure",
          "monthly_charges",
          "support_calls",
          "contract",
          "payment_method"
        ],
        "numeric": [
          "tenure",
          "monthly_charges",
          "support_calls"
        ],
        "categorical": [
          "contract",
          "payment_method"
        ],
        "training": "The host fits on training data only. Do not load files, train, or evaluate at module import."
      },
      "algorithms": [
        {
          "id": "forest",
          "name": "随机森林",
          "rationale": "与线性模型比较非线性关系和特征交互；限制树深度与并行度。",
          "hyperparameter_defaults": {
            "n_estimators": 100,
            "max_depth": 7,
            "min_samples_leaf": 5,
            "class_weight": "balanced"
          },
          "hyperparameter_space": {
            "n_estimators": [
              "int",
              20,
              200
            ],
            "max_depth": [
              "int",
              2,
              12
            ],
            "min_samples_leaf": [
              "int",
              1,
              20
            ],
            "class_weight": [
              null,
              "balanced"
            ]
          }
        },
        {
          "id": "logistic",
          "name": "逻辑回归",
          "rationale": "可解释线性基线；标准化数值特征，使用类别权重处理不平衡。",
          "hyperparameter_defaults": {
            "C": 1.0,
            "max_iter": 1000,
            "class_weight": "balanced",
            "solver": "lbfgs"
          },
          "hyperparameter_space": {
            "C": [
              "float",
              0.01,
              100
            ],
            "max_iter": [
              "int",
              100,
              3000
            ],
            "class_weight": [
              null,
              "balanced"
            ],
            "solver": [
              "lbfgs",
              "liblinear"
            ]
          }
        }
      ],
      "config": "configs/churn.json"
    },
    "text_classification": {
      "target": "label",
      "features": [
        "text"
      ],
      "interface": {
        "build_model": "build_model(seed=42) returns an unfitted sklearn Pipeline",
        "predict": "predict(model, frame) returns 1D finite class-1 probabilities in row order",
        "features": [
          "text"
        ],
        "target": "label",
        "classes": {
          "0": "ordinary",
          "1": "urgent"
        },
        "empty": "Treat missing/empty text as empty string, reject absent text column with ValueError",
        "preprocessing": "Fit character-level TF-IDF on training data only. Handle unseen vocabulary."
      },
      "algorithms": [
        {
          "id": "text_logistic",
          "name": "文本逻辑回归",
          "rationale": "字符 TF-IDF 与线性模型，兼容中文工单。",
          "defaults": {
            "C": 1.0,
            "max_iter": 1000,
            "class_weight": "balanced"
          },
          "parameter_space": {
            "C": [
              "float",
              0.01,
              100
            ],
            "max_iter": [
              "int",
              100,
              3000
            ],
            "class_weight": [
              null,
              "balanced"
            ]
          }
        },
        {
          "id": "text_nb",
          "name": "多项式朴素贝叶斯",
          "rationale": "非负 TF-IDF 上的轻量分类基线，与线性模型比较。",
          "defaults": {
            "alpha": 1.0,
            "fit_prior": true
          },
          "parameter_space": {
            "alpha": [
              "float",
              0.01,
              10
            ],
            "fit_prior": [
              true,
              false
            ]
          }
        }
      ],
      "config": "configs/text.json"
    }
  },
  "metrics": [
    "roc_auc",
    "f1",
    "accuracy",
    "balanced_accuracy"
  ]
}
```
