import os
import time
from multiprocessing import Pool
import tqdm

pbar = tqdm.tqdm(total=100)


def function(val):
    pbar.update(1)
    time.sleep(0.5)
    return [-1 * val]


if __name__ == '__main__':
    print(os.cpu_count())
    pool = Pool()                         # Create a multiprocessing Pool
    out = pool.map(function, range(1000))
    print(out)
