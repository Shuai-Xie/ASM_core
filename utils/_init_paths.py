import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.dirname(__file__)
add_path(this_dir)

modules = ['asm', 'datasets', 'net', 'tools', 'utils']

for m in modules:
    m_path = osp.join(this_dir, '..', m)
    add_path(m_path)
