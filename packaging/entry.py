"""Frozen-application entry point for PyInstaller builds.

PyInstaller freezes a *script*, not a module, so it cannot use
``rfnoise/__main__.py`` directly -- that file relies on a relative import
(``from .cli import main``) which only works when executed as part of the
``rfnoise`` package. This shim uses an absolute import instead, so it behaves
identically whether run from source or from inside a onefile bundle.

The resulting executable is a thin wrapper around :func:`rfnoise.cli.main`, so
every subcommand (``run``, ``ui``, ``gui``, ``list-devices``) works exactly as
it does via the ``rfnoise`` console script.
"""

import sys

from rfnoise.cli import main

if __name__ == "__main__":
    sys.exit(main())
