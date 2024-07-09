import time
from slim.initializers.initializers import rhh, rhh_para
from slim.utils.utils import get_terminals
from datasets.data_loader import load_ppb
from slim.config.gp_config import gp_pi_init
import torch


import numpy as np
import random
normal_speeds, parallel_speed = [], []
X_train = load_ppb(X_y=True)
t = get_terminals(X_train)
f = gp_pi_init["FUNCTIONS"]
c = gp_pi_init["CONSTANTS"]

print("what")

init_pop_size, init_depth, FUNCTIONS, TERMINALS, CONSTANTS, p_c = 200, 8, f, t, c ,0

for seed in range(10):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    start = time.time()

    for _ in range(100):
        results = []
        results.extend(rhh(init_pop_size, init_depth, FUNCTIONS, TERMINALS, CONSTANTS, p_c))

    end = time.time()

    normal_speeds.append(end-start)

    start = time.time()

    for _ in range(100):
        results = []
        results.extend(rhh_para(init_pop_size, init_depth, FUNCTIONS, TERMINALS, CONSTANTS, p_c))

    end = time.time()

    parallel_speed.append(end-start)

print(f'Average normal speed {sum(normal_speeds) / 10}')
print(f'Average inner parallelized speed {sum(parallel_speed) / 10}')