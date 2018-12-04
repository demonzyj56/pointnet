#!/usr/bin/env python
import os
import sys


def build_module(path):
    """Build the module contained in path with current python interpreter."""
    current_path = os.getcwd()
    os.chdir(path)
    assert os.path.exists('setup.py'), 'No setup.py exists in {}'.format(path)
    assert os.system('rm *.so') == 0
    if os.path.exists('build'):
        assert os.system('rm -r build/') == 0
    ret = os.system('{} setup.py build_ext --inplace'.format(sys.executable))
    os.chdir(current_path)
    if ret == 0:
        print('Built modules in {}'.format(path))
    else:
        print('Error building modules in {}'.format(path))
        sys.exit(ret)


def valid_dir(path):
    """Check whether the given path is a valid directory."""
    if not os.path.isdir(path):
        return False
    basename = os.path.basename(path)
    if basename == '__pycache__' or basename.startswith('.'):
        return False
    return True


if __name__ == '__main__':
    module_path = os.path.join(os.path.dirname(__file__), 'modules')
    modules = [n for n in os.listdir(module_path) if
               valid_dir(os.path.join(module_path, n))]
    for m in modules:
        build_module(os.path.join(module_path, m))
