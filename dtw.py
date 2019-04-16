# %%
import numpy as np
import unittest


def DTW(ts_1, ts_2, d=lambda x, y: np.abs(x-y)):
    '''動的時間伸縮法（Dynamic Time Warping）
    時系列データ１と時系列データ２の類似度（距離）を計算する。

    Parameters
    ----------
    ts_1 : numpy.array
        時系列データ１
    ts_2 : numpy.array
        時系列データ２
    d : int
        動的時間伸縮法を実行する際に使用する距離関数

    Returns
    -------
    distance : float
        時系列データ１と時系列データ２の類似度（距離）。
    '''

    # 配列の長さを取得
    ts_1_len = len(ts_1)
    ts_2_len = len(ts_2)
    # コスト行列 (ts_1 と ts_2 のある2点間の距離を保存)
    cost = np.empty(shape=(ts_1_len, ts_2_len))
    # 距離行列 (ts_1 と ts_2 の最短距離を保存)
    dist = np.empty(shape=(ts_1_len, ts_2_len))

    cost[0, 0] = d(ts_1[0], ts_2[0])
    dist[0, 0] = cost[0, 0]

    for i in range(1, ts_1_len):
        cost[i, 0] = d(ts_1[i], ts_2[0])
        dist[i, 0] = dist[i-1, 0] + cost[i, 0]

    for j in range(1, ts_2_len):
        cost[0, j] = d(ts_1[0], ts_2[j])
        dist[0, j] = dist[0, j-1] + cost[0, j]

    window = max(ts_1_len, ts_2_len)
    for i in range(1, ts_1_len):
        window_start = max(1, i-window)
        window_end = min(ts_2_len, i+window)
        for j in range(window_start, window_end):
            choices = np.array([dist[i-1, j], dist[i, j-1], dist[i-1, j-1]])
            cost[i, j] = d(ts_1[i], ts_2[j])
            dist[i, j] = min(choices) + cost[i, j]

    distance = dist[ts_1_len-1, ts_2_len-1]
    return distance


class TestDTW(unittest.TestCase):
    def test_distance(self):
        # import pandas as pd
        # https://www.analyticsvidhya.com/wp-content/uploads/2016/02/AirPassengers.csv
        # def dateparse(dates): return pd.datetime.strptime(dates, '%Y-%m')
        # data = pd.read_csv('./AirPassengers.csv', index_col='Month',
        #                    date_parser=dateparse, dtype='float')
        # ts_1 = data['#Passengers'].values[30:45]
        # ts_2 = data['#Passengers'].values[40:55]

        ts_1 = np.array([199., 199., 184., 162., 146., 166., 171., 180., 193., 181., 183.,
                         218., 230., 242., 209.])
        ts_2 = np.array([183., 218., 230., 242., 209., 191., 172., 194., 196., 196., 236.,
                         235., 229., 243., 264.])

        distance = DTW(ts_1, ts_2)

        self.assertEqual(distance, 286.0, "distance is 280.0")


if __name__ == '__main__':
    unittest.main()
