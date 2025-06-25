import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

# 加载数据
df = pd.read_csv("hand_digits_dataset-[0-9].csv")

# 检查数据分布
print("标签分布:\n", df['label'].value_counts())

# 移除最后一列（角度特征）
X = df.drop("label", axis=1).iloc[:, :-1]  # 关键修改，去除角度
y = df["label"]

# 数据标准化 (SVM对特征缩放敏感)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练/测试集 (保持类别平衡)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)


# 创建SVM模型
def train_svm_model(X_train, y_train):
    # 基础模型
    svm = SVC(kernel='rbf', random_state=42, probability=True)

    # 超参数网格 (可根据需要调整)
    param_grid = {
        'C': [0.1, 1, 10, 100],  # 正则化参数
        'gamma': ['scale', 'auto', 0.1, 1],  # 核函数系数
        'class_weight': [None, 'balanced']  # 处理类别不平衡
    }

    # 使用网格搜索寻找最佳参数
    grid_search = GridSearchCV(
        svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)

    print("\n最佳参数:", grid_search.best_params_)
    return grid_search.best_estimator_


# 训练模型
print("\n开始训练SVM模型...")
model = train_svm_model(X_train, y_train)

# 评估模型
y_pred = model.predict(X_test)
print("\n模型评估:")
print(classification_report(y_test, y_pred))
print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")

# 保存模型和标准化器
joblib.dump(model, "hand_digits_model.pkl")
joblib.dump(scaler, "svm_scaler.pkl")
print("\n模型已保存为 hand_digits_model.pkl")
print("标准化器已保存为 svm_scaler.pkl")

# 打印特征重要性 (SVM可以通过系数分析)
if hasattr(model, 'coef_'):
    print("\n特征重要性(绝对值):")
    importance = np.abs(model.coef_[0])
    for i, imp in enumerate(importance):
        print(f"特征 {i}: {imp:.4f}")