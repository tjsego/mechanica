# """
# Defines installation-specific relative paths. 

# All paths are defined with respect to the mechanica python module root directory
# """
import os
import sys

mx_dir_bin_rel = @MXPY_RPATH_BIN_SITE@
"""Path to mechanica installation bin directory"""

mx_dir_root = os.path.dirname(os.path.abspath(__file__))
mx_dir_bin = None
if mx_dir_bin_rel is not None:
    mx_dir_bin = os.path.abspath(os.path.join(mx_dir_root, mx_dir_bin_rel))

if sys.platform.startswith('win'):
    if mx_dir_bin is not None:
        try:
            env_str = os.environ['PATH']
        except KeyError:
            env_str = ''
        env_str_list = env_str.split(';')

        if mx_dir_bin not in env_str_list:
            os.environ['PATH'] = ';'.join([mx_dir_bin] + env_str_list)
