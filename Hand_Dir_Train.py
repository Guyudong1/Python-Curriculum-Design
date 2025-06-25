import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 加载数据
df = pd.read_csv("hand_direction_dataset.csv")

# 拆分特征和标签
X = df.drop("label", axis=1).values
y = df["label"].values

# 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 训练模型
model = RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42)
model.fit(X_train, y_train)

# 评估
y_pred = model.predict(X_test)
print("\n模型评估:")
print(classification_report(y_test, y_pred))
print(f"准确率: {accuracy_score(y_test, y_pred):.2f}")

# 保存模型
joblib.dump(model, "hand_direction_model.pkl")
print("模型已保存为 hand_direction_model.pkl")
