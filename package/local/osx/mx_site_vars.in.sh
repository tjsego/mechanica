#!/bin/bash
this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" ; pwd -P )"
export MXPYSITEDIR=${this_dir}/@MX_SITEPACKAGES_REL@
export MXENV=${this_dir}/@PY_ROOT_DIR_REL@
