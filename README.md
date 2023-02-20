# DBP391 Project 
This is the implementation of our thesis for Artificial Intelligence Bachelor at FPT University:


## Project member:
- Nguyen Bao Phuoc
- Duong Thuy Trang
- Nguyen Thanh Tung
- Instructor: Phan Duy Hung

## Requirement

Create virtual enviroment and download necessary library list in requirements.txt:
```
python -m venv venv
cd venv/Scripts
activate
cd ../..
pip install requirements.txt
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