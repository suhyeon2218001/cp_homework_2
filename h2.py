# 필요한 라이브러리 임포트
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, confusion_matrix, precision_recall_fscore_support

# Matplotlib 백엔드 설정 초기화
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# 데이터 파일 경로
filename = "C:\\Users\\home\\Downloads\\water_potability\\water_potability.csv"

# 컬럼 이름 정의
column_names = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity', 'Potability']

# 데이터 읽어오기
data = pd.read_csv(filename, names=column_names)

# 데이터 셋의 행렬 크기(shape)
print("데이터 셋의 행렬 크기:", data.shape)

# 데이터 셋의 요약(describe())
print("\n데이터 셋의 요약:")
print(data.describe())

# 결측값 제거 (만약 결측값이 있다면)
data = data.dropna()

# 독립 변수 X와 종속 변수 Y로 나누기
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 시각화 그래프 저장
plt.figure(figsize=(10, 6))
for i in range(X.shape[1]):
    plt.scatter(X[:, i], y, alpha=0.5, label=column_names[i])
plt.legend()
plt.title('Feature vs Potability')
plt.xlabel('Feature Values')
plt.ylabel('Potability')
plt.savefig('feature_vs_potability.png')

# 시각화 그래프 열기
plt.show("C:\\Users\\Documents\\Github\\cp_6\\cp_7\\cp_homework_2\\feature_vs_potability.png")

# 데이터 분할 (학습 데이터와 테스트 데이터)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 전처리: 표준화(Standardization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 머신러닝 모델 생성 및 학습
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 학습된 모델을 사용하여 테스트 데이터에 대한 예측 수행
y_pred = model.predict(X_test)

# 성능 평가
accuracy = accuracy_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

# 결과 출력
print("\n성능 평가:")
print("정확도 (Accuracy):", accuracy)
print("평균 제곱 오차 (MSE):", mse)
print("평균 절대 오차 (MAE):", mae)
print("오차 행렬 (Confusion Matrix):\n", conf_matrix)
print("정밀도 (Precision):", precision)
print("재현율 (Recall):", recall)
print("F1 스코어 (F1 Score):", f1)


