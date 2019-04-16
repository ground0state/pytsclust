import numpy as np
from kmedoids import KMedoids
from matplotlib import pyplot as plt

# 各グループの系列数
N = 20
# 系列の長さ
SPAN = 24
# トレンドが上昇/ 下降する時の平均値
TREND = 0.8

# 時系列データ生成
x = np.array(range(SPAN))
features = np.zeros((0, SPAN))
for i in range(N):
    r = np.random.normal(TREND, 1, size=SPAN)
    y = np.cumsum(r)
    features = np.r_[features, y.reshape(1, -1)]
    plt.plot(x, y, color='C0')
for i in range(N):
    r = np.random.normal(0, 1, size=SPAN)
    y = np.cumsum(r)
    features = np.r_[features, y.reshape(1, -1)]
    plt.plot(x, y, color='C1')
for i in range(N):
    r = np.random.normal(-TREND, 1, size=SPAN)
    y = np.cumsum(r)
    features = np.r_[features, y.reshape(1, -1)]
    plt.plot(x, y, color='C2')
plt.show()

# クラスタ数
N_CLUSTERS = 3
# クラスタリングする
cls = KMedoids(n_clusters=N_CLUSTERS)
pred = cls.fit_predict(features, randomstate=7)

# 各要素をラベルごとに色付けして表示する
cmap = ['C0', 'C1', 'C2', 'C3', 'C4']
for i in range(N_CLUSTERS):
    labels = features[pred == i]
    for y in labels:
        plt.plot(x, y, color=cmap[i])

plt.show()
print(pred)
