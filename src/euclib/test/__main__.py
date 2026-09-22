# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/__main__.py
'''Runs the ``euclib`` unit tests via ``python -m euclib.test``.'''

# Dependencies ###############################################################

from __future__ import annotations


# Testing ####################################################################

def run_tests(verbosity=2, **kwargs):
    '''Runs every ``euclib`` unit test.

    Parameters
    ----------
    verbosity : int, optional
        The verbosity passed to the test runner. The default is ``2``.
    **kwargs
        Additional keyword arguments passed to ``unittest.main``.

    Returns
    -------
    unittest.result.TestResult
        The result of the test run.
    '''
    from unittest import main
    return main('euclib.test', verbosity=verbosity, exit=False, **kwargs)


if __name__ == '__main__':
    run_tests()
