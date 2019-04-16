from sklearn import datasets
import numpy as np
from dtw import DTW
import sys
import unittest


class KMedoids(object):
    """KMedoids 法でクラスタリングするクラス"""

    def __init__(self, n_clusters=2, max_iter=300):
        '''コンストラクタ
        Parameters
        ----------
        n_clusters: int
            クラスタ数
        max_iter : int
            最大イテレーション数
        '''

        self.n_clusters = n_clusters
        self.max_iter = max_iter

        self.cluster_medoids_args_ = None

    def _dtw_distance(self, ts_1, ts_2):
        return DTW(ts_1, ts_2)

    def _print_meaasage(self, _, clusters_index):
        print('iterations:', _)
        print('clusters:', int(len(clusters_index)))

    def fit_predict(self, features, randomstate=0):
        """クラスタリングを実施する
        Args:
            features (numpy.ndarray): ラベル付けするデータ

        Returns:
            numpy.ndarray: ラベルデータ
        """
        '''クラスタリングを実施
        Parameters
        ----------
        features: numpy.ndarray
            データ。
        randomstate : int
            最大イテレーション数

        Returns
        -------
        pred : numpy.ndarray
            ラベル。
        '''

        # ランダムにラベリング
        np.random.seed(randomstate)
        len_features = len(features)
        pred = np.random.randint(0, self.n_clusters, len_features)

        # 距離行列
        distance_matrix = np.zeros((len_features, len_features))
        for i in range(len_features):
            for j in range(i+1, len_features):
                distance_matrix[i][j] = self._dtw_distance(
                    features[i], features[j])
        distance_matrix = distance_matrix + distance_matrix.T
        clusters_index = range(self.n_clusters)

        # クラスタリングをアップデートする
        for _ in range(self.max_iter):
            # 各クラスタごとにメドイドを計算する
            cluster_medoids_args_ = np.array([], dtype=int)
            for i in clusters_index:
                cluster_matrix = distance_matrix[pred == i][:, pred == i]
                min_args = np.argmin(cluster_matrix.sum(axis=0))
                features_args = np.argwhere(pred == i)[min_args]
                cluster_medoids_args_ = np.append(
                    cluster_medoids_args_, features_args)
            self.cluster_medoids_args_ = cluster_medoids_args_

            # 各特徴ベクトルから最短距離となるセントロイドを基準に新しいラベルをつける
            new_pred = np.array([
                np.array([
                    distance_matrix[p, medoids_args]
                    for medoids_args in self.cluster_medoids_args_
                ]).argmin()
                for p in range(len_features)
            ])

            if np.all(new_pred == pred):
                # 更新前と内容を比較して、もし同じなら終了
                break

            pred = new_pred
            # クラスタ数減少に対応
            clusters_index = np.unique(pred)

        self._print_meaasage(_, clusters_index)
        return pred


class TestKMedoids(unittest.TestCase):
    def test_clustering(self):
        # クラスタ数
        N_CLUSTERS = 5
        # Blob データを生成する
        dataset = datasets.make_blobs(centers=N_CLUSTERS)
        # 特徴データ
        features = dataset[0]
        # クラスタリングする
        cls = KMedoids(n_clusters=N_CLUSTERS)

        def test_distance(ts_1, ts_2):
            return 1
        cls._dtw_distance = test_distance

        pred = cls.fit_predict(features)

        self.assertEqual(len(np.unique(pred)), 5, "cluster_num is 5")


if __name__ == '__main__':
    unittest.main()
