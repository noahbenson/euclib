# -*- coding: utf-8 -*-
###############################################################################
# euclib/_version.py
'''Exposes the installed version of ``euclib`` as an ``immlib.Version``.'''

# Dependencies ###############################################################

from __future__ import annotations

from pathlib import Path

from immlib import Version


# Version ####################################################################

euclib_path = Path(__file__).parent
pyproject_path = euclib_path.parent.parent / 'pyproject.toml'

version = Version(package_name=__package__, pyproject_path=pyproject_path)
