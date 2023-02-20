# THESIS
This is the implementation of our thesis for Artificial Intelligence Bachelor at FPT University:


## Project member:
- Nguyen Bao Phuoc
- Duong Thuy Trang
- Nguyen Thanh Tung
- Instructor: Phan Duy Hung

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

To training leiden clustering GCN, run:

```
python main.py --cluster_type leiden
```

*Note: See more hyperparameter in main.py to tuning model if you want*

## Reference
*In process*