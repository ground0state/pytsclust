# from sklearn import datasets
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
        # dataset = datasets.make_blobs(centers=N_CLUSTERS)
        dataset = (np.array([[ -8.70314048,   9.57833679],
       [ -1.36271237,   3.22979878],
       [ -1.93756122,   3.03560482],
       [ -4.69772114,  -8.40889469],
       [ -5.74384571,  -7.60737331],
       [ -6.09942479,  -7.88022696],
       [ -8.66511761,   8.80687734],
       [ -7.80980071,  10.0830379 ],
       [ -4.92527375,   5.75419589],
       [ -6.5801394 ,   5.68151201],
       [ -5.81637129,   6.2221703 ],
       [ -5.17958018,  -6.10984401],
       [ -0.51955337,   0.29400015],
       [ -3.16317579,  -3.63521228],
       [ -6.15970318,   4.92637705],
       [ -6.06685743,  -7.46160338],
       [ -1.36114916,   1.64971389],
       [ -6.56186549,   4.31792703],
       [ -6.85208578,   4.79619302],
       [ -1.03887832,   2.59051649],
       [ -3.39018916,  -2.61049782],
       [  0.36181274,   1.05497792],
       [ -3.59406709,  -1.23338933],
       [ -0.5072758 ,   0.30200174],
       [ -9.73448613,   8.25932538],
       [ -2.54838761,  -2.40784405],
       [ -5.02772896,  -8.06313714],
       [ -5.42472145,   6.16852301],
       [ -2.59738685,  -2.13244694],
       [ -5.38442665,   6.40804369],
       [  0.29086563,   0.92122509],
       [ -1.81654309,  -1.54041117],
       [ -6.07481198,  -8.56738671],
       [ -8.96542293,   9.48966585],
       [  0.91180487,   2.71750586],
       [ -2.62781697,  -2.58475634],
       [ -4.77455973,  -1.8637485 ],
       [ -1.43733697,  -2.8007805 ],
       [ -4.14160236,  -2.8727206 ],
       [ -8.39751341,   9.50922914],
       [  0.90810723,   1.44090674],
       [-10.14280461,   9.4595853 ],
       [ -9.1233923 ,   9.66679992],
       [ -4.00821483,   5.9869324 ],
       [ -8.10774794,   7.59521455],
       [ -2.99126493,  -1.94021579],
       [ -3.81501938,  -1.38404025],
       [  0.76807764,   1.02679351],
       [-10.19640272,   9.3169213 ],
       [ -8.43956159,   9.09659953],
       [ -2.56862036,  -0.66593704],
       [ -7.71888647,   4.96205046],
       [  0.01864845,   1.02333941],
       [ -3.33969704,  -2.18910998],
       [ -4.51639573,  -7.49339112],
       [ -0.53205751,   0.30671326],
       [ -1.94811728,  -5.25228123],
       [ -6.3215496 ,  -7.90076907],
       [ -5.87915041,   4.48769311],
       [ -5.01369774,  -8.03075157],
       [ -6.01398522,  -6.45825359],
       [ -7.25339713,   6.08921423],
       [ -5.13915581,   4.56725411],
       [ -9.4955942 ,   9.02808327],
       [ -6.441235  ,  -8.55664354],
       [ -5.2899334 ,  -6.25594344],
       [ -8.80840598,   9.64463342],
       [-11.13940034,   9.63504707],
       [ -8.07724002,   8.2276517 ],
       [ -0.50388923,   2.2849383 ],
       [ -5.71614067,   7.79631702],
       [ -1.00625048,  -0.90995244],
       [ -5.68620079,   4.4805684 ],
       [ -4.47394294,  -6.88333718],
       [ -6.4961548 ,   6.32971486],
       [ -6.17094364,  -7.33936882],
       [ -4.64894785,  -8.04975357],
       [ -3.13695238,  -2.28708351],
       [ -8.74825181,  10.64041093],
       [  0.46207804,   3.22105714],
       [ -5.70391708,  -6.57140746],
       [ -1.37790497,   2.6835331 ],
       [  0.01599481,   0.84997957],
       [ -4.80396901,  -5.99357802],
       [ -3.17498631,  -4.32671263],
       [ -2.3470367 ,  -2.69151929],
       [ -5.90284737,  -8.52921879],
       [ -4.75665135,   5.12018194],
       [ -3.26978527,  -3.5084195 ],
       [ -8.01966054,   8.60836489],
       [ -7.82911138,   5.91044364],
       [-10.33653833,   8.00163476],
       [ -4.17910255,  -1.28407732],
       [ -6.85991485,  -6.50765729],
       [ -5.67500943,   4.2451996 ],
       [ -7.06562994,   3.88239898],
       [ -1.58433111,   3.1918266 ],
       [  0.77488166,   1.74224621],
       [ -5.94253258,   4.01836587],
       [ -7.74647368,   8.88729677]]), np.array([1, 3, 3, 2, 2, 2, 1, 1, 0, 0, 0, 2, 3, 4, 0, 2, 3, 0, 0, 3, 4, 3,
       4, 3, 1, 4, 2, 0, 4, 0, 3, 4, 2, 1, 3, 4, 4, 4, 4, 1, 3, 1, 1, 0,
       1, 4, 4, 3, 1, 1, 4, 0, 3, 4, 2, 3, 4, 2, 0, 2, 2, 0, 0, 1, 2, 2,
       1, 1, 1, 3, 1, 3, 0, 2, 0, 2, 2, 4, 1, 3, 2, 3, 3, 2, 4, 4, 2, 0,
       4, 1, 0, 1, 4, 2, 0, 0, 3, 3, 0, 1]))
        print(dataset)
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
