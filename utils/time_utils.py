import functools
import time
from datetime import datetime


# 显示程序执行时间
def exe_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        func(*args, **kwargs)
        t2 = time.time()
        print('exe time:', t2 - t1)

    return wrapper


def get_curtime():
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    return current_time
