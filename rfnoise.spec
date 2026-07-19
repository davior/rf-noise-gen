# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build spec for rfnoise -- one cross-platform onefile binary.

Run with::

    pyinstaller rfnoise.spec

The output is ``dist/rfnoise`` (``dist/rfnoise.exe`` on Windows): a single,
self-contained executable that needs no Python install.

Why this spec exists (and isn't just ``pyinstaller packaging/entry.py``):
rfnoise's three optional dependencies are all imported *lazily* so the core
stays dependency-free --

    * numpy      -> rfnoise/modulation.py   (AM/FM/chirp DSP)
    * pyserial   -> rfnoise/devices/tinysa.py
    * dearpygui  -> rfnoise/gui.py          (graphical editor)

PyInstaller discovers imports by static analysis, which cannot see an
``import`` that lives inside a function body. Left to itself it would ship a
binary that crashes the moment a user runs ``rfnoise gui`` or asks for
modulation. So we force-collect all three here.

Built in **console** mode on purpose: rfnoise is CLI/TUI-first (``run``,
``list-devices``, the default text ``ui``) and needs stdout/stderr. ``gui``
still opens its own native OpenGL window from a console binary, so nothing is
lost by keeping the console.
"""

from PyInstaller.utils.hooks import collect_all, collect_submodules

# Dear PyGui ships a compiled extension (with its windowing/OpenGL statically
# linked). collect_all grabs that binary plus any package data and submodules.
dpg_datas, dpg_binaries, dpg_hiddenimports = collect_all("dearpygui")

# pyserial selects its backend at runtime (serialposix / serialwin32 / ...),
# and the port lister lives in serial.tools -- collect every submodule so the
# right one is present on each OS.
serial_hiddenimports = collect_submodules("serial")

hiddenimports = (
    dpg_hiddenimports
    + serial_hiddenimports
    + [
        # numpy's bundled PyInstaller hook collects its binaries/data once it
        # sees numpy referenced; listing it here is what triggers that.
        "numpy",
    ]
)

a = Analysis(
    ["packaging/entry.py"],
    pathex=[SPECPATH],  # noqa: F821 -- SPECPATH is injected by PyInstaller
    binaries=dpg_binaries,
    datas=dpg_datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest", "_pytest"],  # test-only; never needed at runtime
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="rfnoise",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    # UPX left off deliberately: it can corrupt numpy/dearpygui native
    # libraries and trips antivirus false positives on Windows.
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,  # host arch per runner (Intel vs Apple Silicon covered in CI)
    codesign_identity=None,
    entitlements_file=None,
)
