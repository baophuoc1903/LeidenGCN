import sys
import os
filename = r"F:\Thesis\DeeperGCN\deep_gcns_torch-master\examples\ogb\ogbn_proteins\__init__.py"
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(filename)))))
sys.path.insert(0, ROOT_DIR)
