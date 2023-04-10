# THESIS
This is the implementation of our thesis for Artificial Intelligence Bachelor at FPT University:

[Improving Cluster-GCN with additional constraints](https://docs.google.com/document/d/13TfwmjR6ge3QBtAgvVNVGU5nWnMjiD136a0UwaTbkFQ/edit?usp=sharing)

## Project member:
- Nguyen Bao Phuoc
- Duong Thuy Trang
- Nguyen Thanh Tung
- Instructor: Associate Professor Dr. Phan Duy Hung

## Requirement

Create virtual enviroment:
```
python -m venv venv
cd venv/Scripts
activate
cd ../..
```

Download necessary library:
```
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip install torch-geometric
pip install torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip install -r requirements.txt
```

## Training

To training random clustering GCN, run:

```
python main.py
```

To training leiden clustering GCN with min/max constraints, run:

```
python main.py --cluster_type leiden
```

To training leiden clustering GCN with min/max constraints and overlaping constraint, run:
```
python main.py --cluster_type leiden_species --overlap_ratio 0.5 --edge_drop 0.5 --clusters_path dataset/leiden_clusters_80_100_weight_final.csv
```

*Note: See more hyperparameter in main.py to tuning model if you want*

## Reference
*List in our thesis document*