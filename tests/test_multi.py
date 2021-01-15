import os
import tqdm
import time
from multiprocessing import Pool


class Test:

    def __init__(self):
        self.pbar = None
        self.pool = Pool()  # Create a multiprocessing Pool

    def func(self, i):
        time.sleep(0.2)
        self.pbar.update(1)
        return [-1 * i]

    def main(self, _list):
        self.pbar = tqdm.tqdm(total=int(len(_list) / os.cpu_count()))

        print('CPU count: {}'.format(os.cpu_count()))

        out = self.pool.map(self.func, _list)
        print(out)


if __name__ == '__main__':
    t = Test()
    t.main(list(range(1000)))
