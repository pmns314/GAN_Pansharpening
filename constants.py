import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(ROOT_DIR, "datasets/")
EPS = 1e-12
TO_SAVE = [1, 2, 3, 5, 8,
           10, 20, 30, 50, 80,
           100, 200, 300, 500, 800,
           1000, 2000, 3000, 5000, 8000, 10000]
L = 11
Qblocks_size = 32
flag_cut_bounds = 1
dim_cut = 21
th_values = 0
ratio = 4.0
