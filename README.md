# Gated GAT for short-term traffic forecasting

TensorFlow implementation accompanying:

> Yue Li, Shujuan Chen, Ying Jin. *Explaining spatial information flow in short-term traffic
> forecasting models using a gated graph attention network.*

A gate is inserted at the node update of each graph attention (GAT) layer of ST-MetaNet. For every
sensor and every time step the gate returns the share of the node update taken from neighbouring
sensors rather than from the sensor itself. An L1 penalty on the gate withdraws neighbour
information progressively, which gives a graded ablation whose conclusions are then checked against
deleting the GAT layers outright and retraining.

Historical note on naming: the mechanism was called the Dynamic Central Update (DCU) during
development, and `DCU` still appears in directory names, saved weights and log filenames. It is the
same thing the paper calls the gate.

---

## Repository layout

| Path | Contents |
|---|---|
| `ST-MetaNet/` | baseline, fixed scalar gate |
| `ST-MetaNet-DCU/` | gated GAT, no penalty |
| `ST-MetaNet-DCU-regularization/` | gated GAT with the L1 penalty on the gate |
| `data/` | traffic data, edge and node features |
| `param/` | trained weights, one `.h5` per penalty strength |
| `eval/` | training and evaluation logs, one per run |
| `interpret/rho-matrix.ipynb` | loads trained weights and extracts the gate matrix |

The three model folders differ in `graph.py`, where `MetaGAT` is modified, and additionally in
`seq2seq.py` and `train.py` for the regularised variant, where the gate values are returned
alongside the prediction and enter the loss.

## Requirements

Python 3.9.18 on Windows 11. See `requirements.txt`. DGL must be switched to the TensorFlow
backend after installation. Manual installation of the CUDA components is recommended.

## Data

Unzip `data/DATA.zip` and place both files in the same directory.

- `sensor498_2019-01-01_2019-12-31.h5`, as downloaded from
  [National Highways TRIS](http://tris.highwaysengland.co.uk/), shape (498, 35040, 2) for
  (sensors, timestamps, features), features being speed and volume.
- `sensor498_2019-01-01_2019-12-31_imputed.h5`, the file the training pipeline actually reads.
  Imputation applies to speed only, where both volume and speed are recorded as 0, and replaces the
  speed with the closest previous non-missing value. It changes 54,025 speed cells and no volume
  cells. The notebook is `data/imputation.ipynb`.
- `tris_edge_features.xlsx`, shape (498, 10). Sensor longitude and latitude, segment distance and
  driving duration, origin and destination longitude and latitude, and the neighbour count at each
  end. Two segments carry no valid sensor and take the reading from the opposite direction, marked
  by a sensor id beginning `999`.
- `tris_node_features.csv`, shape (181, 2), junction longitude and latitude.

Other years, 2015 to 2023, and the download script are linked from the release notes.

## Running

```
cd ST-MetaNet-DCU-regularization
python train.py --file model_setting/st-metanet.yaml --epochs 2
```

`run.ipynb` runs `train.py` repeatedly to sweep the penalty strength. The penalty strength is
`training.lambda_reg` in `model_setting/st-metanet.yaml`.

## What is and is not reproducible here

Please read this before using the repository to check a number in the paper.

- **The penalty sweep is reproducible.** The sixteen penalty strengths in Table 1 come from
  `ST-MetaNet-DCU-regularization` with `lambda_reg` set per run. The corresponding weights are in
  `param/` and the logs in `eval/`.
- **The four ablation variants are not included.** Removing the encoder GAT layer, the decoder GAT
  layer, or both was done by commenting out source in place rather than by a configuration flag, so
  Table 3 cannot be regenerated from this tree as it stands. The resulting logs are in `param/`
  (`st-metanet-DCU-encodergat_removed.log`, `st-metanet-DCU-no_GAT.log`).
- **`lambda_reg` in the shipped yaml is `1000000.0`**, which is not a setting used in the paper. Set
  it explicitly for every run.
- **Units.** The training code records volume as the mean of the constituent 15 minute counts and
  speed as their sum. Volume must be multiplied by 4 for vehicles per hour, and speed divided by the
  number of 15 minute periods in the aggregation interval for miles per hour. Getting this wrong is
  the easiest mistake to make with these logs.
- Every reported model was trained for a fixed number of updates from a single seed (`seed: 2`),
  with no early stopping and no best-checkpoint selection. Comparisons smaller than about 4 vehicles
  per hour are not interpretable from a single run. This is stated in the paper.

## Baseline

For the original MXNet implementation of ST-MetaNet see the
[ST-MetaNet repository](https://github.com/panzheyi/ST-MetaNet) and:

> Zheyi Pan, Yuxuan Liang, Weifeng Wang, Yong Yu, Yu Zheng, Junbo Zhang. Urban Traffic Prediction
> from Spatio-Temporal Data Using Deep Meta Learning. KDD'19.

## License

MIT. See `LICENSE`.
