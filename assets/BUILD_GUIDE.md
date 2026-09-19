# v1 build policy / v1 建置範圍

This v1 maintenance branch supports Python 3.10–3.12 and source-only releases.
The former Win7/Python 3.8 client-only PyInstaller recipe is retired.
`build-client.spec` stops immediately with an explanation; it no longer creates
junctions into a source checkout or an apparently portable client directory.
The inherited `build.spec` is not a qualified v1 portable release workflow.

v1 維護分支支援 Python 3.10–3.12，僅發行原始碼。舊版 Win7／Python 3.8
客戶端打包流程已停用；測試矩陣通過不代表 Windows 可攜式套件已驗證。

For supported source setup, container deployment, version channels and validation:

- [English maintenance policy](../docs/en/maintenance.md)
- [繁體中文維護政策](../docs/zh-TW/maintenance.md)
- [Upstream migration guide](../docs/v1-upstream-refresh.md)
- [Validation evidence and limits](../docs/v1-refresh-validation.md)

For a supported Windows package workflow, use the separate v2 branch and its
`assets/BUILD_GUIDE.md`. Physical microphone, keyboard hooks and foreground text
output still require Windows hardware validation.
