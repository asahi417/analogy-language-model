import os
import tqdm
import time
from multiprocessing import Pool

pbar = tqdm.tqdm()


class Func:

    def __init__(self, c):
        self.c = c

    def func(self, x):
        i, k = x
        time.sleep(0.01)
        pbar.update(1)
        return [self.c * i + k]


class Test:

    def __init__(self):
        pass

    def main(self, _list):
        pool = Pool()  # Create a multiprocessing Pool
        f = Func(-10)
        print('CPU count: {}'.format(os.cpu_count()))
        out = pool.map(f.func, _list)
        print(out)
        pool.close()


if __name__ == '__main__':
    t = Test()
    tmp = [[i, i * -1] for i in range(1000)]
    t.main(tmp)
