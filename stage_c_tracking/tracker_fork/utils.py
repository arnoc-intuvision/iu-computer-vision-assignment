"""Minimal vendored utils from eddyhkchiu/mahalanobis_3d_multi_object_tracking.

Only the functions imported by main.py are kept: mkdir_if_missing, fileparts,
load_list_from_folder. glob2 dependency removed (use glob + recursive walk).
"""

import os
import glob
import numpy as np


def isstring(string_test):
    return isinstance(string_test, str)


def islist(list_test):
    return isinstance(list_test, list)


def islogical(logical_test):
    return isinstance(logical_test, bool)


def isnparray(nparray_test):
    return isinstance(nparray_test, np.ndarray)


def isinteger(integer_test):
    if isnparray(integer_test):
        return False
    try:
        return isinstance(integer_test, int) or int(integer_test) == integer_test
    except (TypeError, ValueError):
        return False


def is_path_valid(pathname):
    try:
        if not isstring(pathname) or not pathname:
            return False
    except TypeError:
        return False
    else:
        return True


def is_path_exists(pathname):
    try:
        return is_path_valid(pathname) and os.path.exists(pathname)
    except OSError:
        return False


def is_path_creatable(pathname):
    if not is_path_valid(pathname):
        return False
    pathname = os.path.normpath(pathname)
    pathname = os.path.dirname(os.path.abspath(pathname))
    while not is_path_exists(pathname):
        pathname_new = os.path.dirname(os.path.abspath(pathname))
        if pathname_new == pathname:
            return False
        pathname = pathname_new
    return os.access(pathname, os.W_OK)


def is_path_exists_or_creatable(pathname):
    try:
        return is_path_exists(pathname) or is_path_creatable(pathname)
    except OSError:
        return False


def isfolder(pathname):
    if is_path_valid(pathname):
        pathname = os.path.normpath(pathname)
        if pathname == './':
            return True
        name = os.path.splitext(os.path.basename(pathname))[0]
        ext = os.path.splitext(pathname)[1]
        return len(name) > 0 and len(ext) == 0
    else:
        return False


def safe_path(input_path, warning=True, debug=True):
    if debug:
        assert isstring(input_path), 'path is not a string: %s' % input_path
    safe_data = os.path.normpath(input_path)
    return safe_data


def fileparts(input_path, warning=True, debug=True):
    good_path = safe_path(input_path, debug=debug)
    if len(good_path) == 0:
        return ('', '', '')
    if good_path[-1] == '/':
        if len(good_path) > 1:
            return (good_path[:-1], '', '')
        else:
            return (good_path, '', '')
    directory = os.path.dirname(os.path.abspath(good_path))
    filename = os.path.splitext(os.path.basename(good_path))[0]
    ext = os.path.splitext(good_path)[1]
    return (directory, filename, ext)


def mkdir_if_missing(input_path, warning=True, debug=True):
    good_path = safe_path(input_path, warning=warning, debug=debug)
    if debug:
        assert is_path_exists_or_creatable(good_path), \
            'input path is not valid or creatable: %s' % good_path
    dirname, _, _ = fileparts(good_path)
    if not is_path_exists(dirname):
        mkdir_if_missing(dirname)
    if isfolder(good_path) and not is_path_exists(good_path):
        os.mkdir(good_path)


def load_list_from_folder(folder_path, ext_filter=None, depth=1,
                           recursive=False, sort=True, save_path=None, debug=True):
    folder_path = safe_path(folder_path)
    if debug:
        assert isfolder(folder_path), 'input folder path is not correct: %s' % folder_path
    if not is_path_exists(folder_path):
        print('the input folder does not exist\n')
        return [], 0

    if isstring(ext_filter):
        ext_filter = [ext_filter]

    fulllist = list()
    if depth is None:
        # recursive glob via os.walk (replaces glob2 dependency)
        for root, dirs, files in os.walk(folder_path):
            if ext_filter is not None:
                for ext_tmp in ext_filter:
                    curlist = [f for f in files if f.endswith(ext_tmp)]
                    curlist = [os.path.join(root, f) for f in curlist]
                    if sort:
                        curlist = sorted(curlist)
                    fulllist += curlist
            else:
                curlist = [os.path.join(root, f) for f in files]
                if sort:
                    curlist = sorted(curlist)
                fulllist += curlist
    else:
        wildcard_prefix = '*'
        for index in range(depth - 1):
            wildcard_prefix = os.path.join(wildcard_prefix, '*')
        if ext_filter is not None:
            for ext_tmp in ext_filter:
                wildcard = wildcard_prefix + ext_tmp
                curlist = glob.glob(os.path.join(folder_path, wildcard))
                if sort:
                    curlist = sorted(curlist)
                fulllist += curlist
        else:
            wildcard = wildcard_prefix
            curlist = glob.glob(os.path.join(folder_path, wildcard))
            if sort:
                curlist = sorted(curlist)
            fulllist += curlist
        if recursive and depth > 1:
            newlist, _ = load_list_from_folder(folder_path=folder_path,
                                               ext_filter=ext_filter,
                                               depth=depth - 1, recursive=True)
            fulllist += newlist

    fulllist = [os.path.normpath(path_tmp) for path_tmp in fulllist]
    num_elem = len(fulllist)

    if save_path is not None:
        save_path = safe_path(save_path)
        if debug:
            assert is_path_exists_or_creatable(save_path), 'the file cannot be created'
        with open(save_path, 'w') as file:
            for item in fulllist:
                file.write('%s\n' % item)

    return fulllist, num_elem