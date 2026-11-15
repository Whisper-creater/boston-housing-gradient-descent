"""
 变量顺序：
 城镇人均犯罪率
 ZN 指面积超过 25,000 平方英尺的住宅用地比例。
 INDUS 城镇非零售商业用地面积比例
 CHAS查尔斯河虚拟变量（如果地块与河流相邻则为1；否则为0）
 NOx 一氧化氮浓度（千万分之一）
 每套住宅的平均房间数
 1940 年前建造的自住房屋的年龄比例
 DIS到波士顿五个就业中心的加权距离
 RAD 指数（放射状公路可达性指数）
 税额：每10,000美元房产的全额房产税税率
 按城镇划分的 PTRATIO 师生比
 B 1000(Bk - 0.63)^2，其中 Bk 为各城镇黑人比例。
 LSTAT % 人口地位较低
 MEDV 指业主自住房屋的中位价值（单位：千美元）
"""
#导包
#from sklearn.datasets import load_boston#由于种族歧视被移除了qvq
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler#特征预处理标准化
from sklearn.model_selection import train_test_split#训练集和测试集的切分
from sklearn.linear_model import LinearRegression#线性回归模型->正规方程
from sklearn.linear_model import SGDRegressor#线性回归模型->梯度下降
from sklearn.metrics import mean_squared_error,mean_absolute_error,root_mean_squared_error
#1.加载数据
data_path = Path(__file__).with_name("boston_house_prices.csv")
housing = pd.read_csv(data_path, encoding="utf-8-sig")
data = housing.drop(columns="自住房屋价值中位数（千美元）").to_numpy()
target = housing["自住房屋价值中位数（千美元）"].to_numpy()


#2.数据预处理->切训练集,测试集
#参1:特征数据 参2:标签数据 参3:测试集所占比例 参4:随机种子(如果种子一致,每次获取到的数据都是固定随机的)
x_train,x_test,y_train,y_test = train_test_split(data,target,test_size=0.2,random_state=17)

#3.特征工程
#3.1创建 标准化对象
transfer=StandardScaler()
#3.2对训练集 测试集 的特征数据 进行标准化
x_train=transfer.fit_transform(x_train)#训练并转换
x_test=transfer.transform(x_test)#转换
#4.模型训练
#4.1创建 线性回归 梯度下降 模型对象
estimator = SGDRegressor()
#4.2模型训练
estimator.fit(x_train,y_train)
#4.3扩展 打印w,b
print(f'w:{estimator.coef_}')
print(f'b:{estimator.intercept_}')
#5.模型预测
y_pred = estimator.predict(x_test)
print(f'预测结果:{y_pred}')

#6.模型评估
#参1真实值 参2模型预测结果
print(f'MAE:{mean_absolute_error(y_test,y_pred)}')
print(f'MSE:{mean_squared_error(y_test,y_pred)}')
print(f'RMSE:{np.sqrt(mean_squared_error(y_test,y_pred))}')

