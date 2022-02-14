"""
This is a temporary supplement to a comprehensive test suite
"""

import mechanica as mx
if not mx.has_cuda:
    raise EnvironmentError
