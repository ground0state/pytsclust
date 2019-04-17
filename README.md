# pytsclust

The package to perform time series clustering.

## Getting Started

### Prerequisites

What things you need to install the software and how to install them,

- NumPy.

### Installing

just modules.

### how to use

The `features` is matrix(numpy.array) and its rows are time series datas.
We specify `N_CLUSTERS` as number of clusters.

```
cls = KMedoids(n_clusters=N_CLUSTERS)
pred = cls.fit_predict(features, randomstate=7)
```

`sample.py` is a sample file. To execute sample you need `matplotlib`.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
