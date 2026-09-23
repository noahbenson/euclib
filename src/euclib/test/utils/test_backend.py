# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/utils/test_backend.py
'''Tests for the backend flags in ``euclib.utils._core``.

The two environment variables are read once, when the module is imported, so a
test of what they do cannot run in the process that is already running: by the
time it asks, the answer has been decided. Each test therefore runs a fresh
interpreter and reads what it reports.

This is worth doing rather than assuming, because these flags are how a user
gets out of a broken build and how a packager insists on a working one, and the
code that reads them has no other test at all.
'''

# Dependencies ###############################################################

from __future__ import annotations

from os import environ
from subprocess import run
from sys import executable
from unittest import TestCase, skipUnless

from euclib.utils._core import using_c_extension


# Running in a fresh interpreter #############################################

#: What the child process prints: whether the extension is in use, and whether
#: the kernels answer correctly.
CHILD = (
    "import numpy as np\n"
    "import euclib\n"
    "from euclib.utils import split_cells, tetrahedron_box_vertices\n"
    "bounds = np.array([[0., 1.], [0., 1.], [0., 1.]])\n"
    "centers = np.array([[0.2, 0.8], [0.2, 0.8], [0.2, 0.8]])\n"
    "tet = np.array([[0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])\n"
    "print('using', bool(euclib.using_c_extension))\n"
    "print('error', euclib.backend_error is None)\n"
    "print('cells', split_cells(centers, bounds).tolist())\n"
    "print('corners', tetrahedron_box_vertices(tet, bounds).shape)\n"
    "print('kernel', 'C' if split_cells.accelerated_here else 'Python')\n"
)


#: The children that have been run, by the settings they were run under.
_RUNS = {}


#: The variables that decide the backend.
FLAGS = ('EUCLIB_NO_C_EXTENSIONS', 'EUCLIB_REQUIRE_C')


def child_environment(**variables):
    '''Builds an environment that asks about one configuration and no other.

    The suite itself may be running under either flag, and those settings are
    inherited by a child process. A child asked what ``EUCLIB_NO_C_EXTENSIONS``
    does has to be asked with ``EUCLIB_REQUIRE_C`` cleared, or the answer would
    be about the pair --- and one of the pair raises rather than answers. Both
    are cleared here and then set as the caller asks, with ``None`` meaning
    "left unset".

    Parameters
    ----------
    variables : dict
        The backend variables to set, and their values.

    Returns
    -------
    dict
        The environment for a child process.
    '''
    environment = dict(environ)
    for name in FLAGS:
        environment.pop(name, None)
    for (name, value) in variables.items():
        if value is None:
            environment.pop(name, None)
        else:
            environment[name] = value
    return environment


def _run(**variables):
    '''Runs ``CHILD`` in a fresh interpreter and returns its output.

    Children are run once per set of settings and remembered, since starting an
    interpreter to import NumPy and SciPy costs a second or so and several tests
    ask about the same settings.

    Parameters
    ----------
    variables : dict
        Environment variables to add to the child's environment.

    Returns
    -------
    (returncode, str)
        The child's exit status and its standard output and error, together.
    '''
    key = tuple(sorted(variables.items()))
    if key not in _RUNS:
        finished = run([executable, '-c', CHILD],
                       env=child_environment(**variables),
                       capture_output=True, text=True)
        _RUNS[key] = (finished.returncode, finished.stdout + finished.stderr)
    return _RUNS[key]


def _report(output, /):
    '''Reads the child's report into a dictionary.'''
    report = {}
    for line in output.splitlines():
        parts = line.split(' ', 1)
        if len(parts) == 2 and parts[0] in ('using', 'error', 'cells',
                                            'corners', 'kernel'):
            report[parts[0]] = parts[1]
    return report


def _state(output, /):
    '''Reads the line a child prints as "state ..." into a list of words.'''
    for line in output.splitlines():
        parts = line.split()
        if parts and parts[0] == 'state':
            return parts[1:]
    return []


# Tests ######################################################################

class TestDisablingTheExtension(TestCase):
    '''``EUCLIB_NO_C_EXTENSIONS``.'''

    def test_the_extension_is_not_loaded(self):
        (status, output) = _run(EUCLIB_NO_C_EXTENSIONS='1')
        self.assertEqual(status, 0, output)
        report = _report(output)
        self.assertEqual(report['using'], 'False')
        self.assertEqual(report['kernel'], 'Python')

    def test_a_reason_is_given(self):
        # A library that quietly ignores its own flag is worse than one that
        # records why it did so.
        (_, output) = _run(EUCLIB_NO_C_EXTENSIONS='1')
        self.assertEqual(_report(output)['error'], 'False')

    def test_the_kernels_still_answer(self):
        (_, output) = _run(EUCLIB_NO_C_EXTENSIONS='1')
        report = _report(output)
        self.assertEqual(report['cells'], '[0, 7]')
        self.assertEqual(report['corners'], '(3, 4)')

    def test_a_false_looking_value_leaves_it_alone(self):
        # The comparison is against a child with the variable unset rather than
        # against this process: this process may itself be running under the
        # flag, and what is being tested is the value's spelling.
        (_, absent) = _run()
        for value in ('0', ''):
            with self.subTest(value=value):
                (_, output) = _run(EUCLIB_NO_C_EXTENSIONS=value)
                self.assertEqual(_report(output)['using'],
                                 _report(absent)['using'])


class TestRequiringTheExtension(TestCase):
    '''``EUCLIB_REQUIRE_C``.'''

    @skipUnless(using_c_extension, "the C extension is not built")
    def test_a_working_extension_satisfies_it(self):
        (status, output) = _run(EUCLIB_REQUIRE_C='1')
        self.assertEqual(status, 0, output)
        self.assertEqual(_report(output)['using'], 'True')

    def test_an_absent_extension_raises(self):
        # The case the flag exists for: a packager who needs to know that the
        # extension was built rather than to be warned that it was not.
        (status, output) = _run(EUCLIB_REQUIRE_C='1',
                                EUCLIB_NO_C_EXTENSIONS='1')
        self.assertNotEqual(status, 0)
        self.assertIn('EUCLIB_REQUIRE_C', output)

    def test_a_false_looking_value_leaves_it_alone(self):
        for value in ('0', ''):
            with self.subTest(value=value):
                (status, output) = _run(EUCLIB_REQUIRE_C=value,
                                        EUCLIB_NO_C_EXTENSIONS='1')
                self.assertEqual(status, 0, output)


class TestTheFlagItself(TestCase):
    '''What this process reports.'''

    def test_using_c_extension_is_a_bool(self):
        self.assertIsInstance(using_c_extension, bool)

    def test_backend_error_agrees_with_it(self):
        # An extension in use has no error behind it, and one that is not in
        # use always has a reason.
        import euclib
        self.assertEqual(euclib.backend_error is None, using_c_extension)

    def test_the_kernel_is_the_one_the_dispatcher_built(self):
        from euclib.utils import _pycore
        from euclib.utils import tetrahedron_box_vertices
        self.assertIs(_pycore._vertices_kernel, tetrahedron_box_vertices)


@skipUnless(using_c_extension, "the C extension is not built")
class TestReloadingReReadsTheFlags(TestCase):
    '''``reload_euclib`` after the flags change.

    ``euclib.utils._core`` decides the backend once, when it is imported.
    Reloading the library is supposed to revisit that decision, which is the
    whole point of having the flag there: someone developing against the
    extension should be able to turn it off and see what the pure-Python
    kernels do, without restarting.

    The reload happens in a child rather than here, because it rebinds every
    class in the library and the tests running alongside this one hold
    references to the old ones.
    '''

    #: Turn the extension off, reload, look at the state, turn it back on.
    CHILD = (
        "import os\n"
        "from euclib.utils import split_cells\n"
        "import euclib\n"
        "def state():\n"
        "    from euclib.utils import split_cells as k\n"
        "    return 'C' if k.accelerated_here else 'Python'\n"
        "first = state()\n"
        "os.environ['EUCLIB_NO_C_EXTENSIONS'] = '1'\n"
        "euclib.reload_euclib()\n"
        "second = state()\n"
        "del os.environ['EUCLIB_NO_C_EXTENSIONS']\n"
        "euclib.reload_euclib()\n"
        "print('state', first, second, state(), bool(euclib.using_c_extension))\n"
    )

    def test_a_reload_picks_up_the_flag(self):
        finished = run([executable, '-c', self.CHILD],
                       env=child_environment(), capture_output=True, text=True)
        output = finished.stdout + finished.stderr
        self.assertEqual(finished.returncode, 0, output)
        self.assertEqual(_state(output), ['C', 'Python', 'C', 'True'])
