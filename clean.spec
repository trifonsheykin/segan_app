# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from pathlib import Path
block_cipher = None


a = Analysis(['clean.py'],
             pathex=['/Users/trifon.sheykin/PythonProjects/ML/segan_exe'],
             binaries=[('/System/Library/Frameworks/Tk.framework/Tk', 'tk'), ('/System/Library/Frameworks/Tcl.framework/Tcl', 'tcl')],
             datas=[ ('config.conf', '.'), ('train.opts', '.') ],
             hiddenimports=['sklearn.utils._cython_blas', 'sklearn.neighbors.typedefs', 'sklearn.neighbors.quad_tree', 'sklearn.tree._utils'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)


# A dependency of libzbar.dylib that PyInstaller does not detect
MISSING_DYLIBS = (
    Path('/Applications/anaconda3/envs/sgn/lib/python3.6/site-packages/torch/lib/libtorch_global_deps.dylib'),
)
a.binaries += TOC([
    (lib.name, str(lib.resolve()), 'BINARY') for lib in MISSING_DYLIBS
])
MISSING_DYLIBS_2 = (
    Path('/Applications/anaconda3/envs/sgn/lib/python3.6/site-packages/samplerate/_samplerate_data/libsamplerate.dylib'), 
)
a.binaries += TOC([
    (lib.name, str(lib.resolve()), 'BINARY') for lib in MISSING_DYLIBS_2
])

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='clean',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=False )
app = BUNDLE(exe,
             name='clean.app',
             icon=None,
             bundle_identifier='store.smartlocks.fl')
