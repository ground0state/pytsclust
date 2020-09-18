# pytsclust

The module to perform time series clustering.

![Figure_1](https://user-images.githubusercontent.com/35659728/56433179-fbfd1200-630a-11e9-9237-eb827153c61d.png)

## Getting Started

### Prerequisites

What things you need to install the software and how to install them,

- NumPy.

### Installing

just a module.

### how to use

The `features` is matrix(numpy.array) and its rows are time series datas.
We specify `N_CLUSTERS` as number of clusters.

```
cls = KMedoids(n_clusters=N_CLUSTERS)
pred = cls.fit_predict(features, randomstate=7)
```

`sample.py` is a sample file. To execute sample you need `matplotlib`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
