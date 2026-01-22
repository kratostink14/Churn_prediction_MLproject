# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['mini_app.py.py'],
    pathex=[],
    binaries=[],
    datas=[('bank_churn_model.pkl', '.'), ('le_gender.pkl', '.'), ('le_geography.pkl', '.'), ('le_card_type.pkl', '.')],
    hiddenimports=['sklearn.ensemble._forest', 'sklearn.utils._cython_blas', 'sklearn.utils._typedefs'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PyQt5', 'PySide6'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='mini_app.py',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
