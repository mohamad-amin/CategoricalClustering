# manual_example.py

import time

from pyprof_timer import Timer, Tree


def main():
    t = Timer('sleep1', parent_name='main').start()
    time.sleep(1)
    t.stop()

    t = Timer('sleep2', parent_name='main').start()
    time.sleep(1.5)
    t.stop()

    t = Timer('sleep1', parent_name='main').start()
    time.sleep(1)
    t.stop()

    t = Timer('sleep2', parent_name='main').start()
    time.sleep(1.5)
    t.stop()

    print(Tree(Timer.root))


if __name__ == '__main__':
    main()
