# coding: utf-8
"""升級不會覆寫的使用者設定（``user/`` 資料夾）。

發行 ZIP 永遠不含 ``user/``，因此把新版解壓覆蓋舊資料夾時，這裡的檔案會原樣保留：

- ``user/config_client.py``、``user/config_server.py``：只寫要改的類別屬性，
  在程式資料夾同名設定檔載入時套用（Server 子行程重新 import 時也會套用）。
- ``user/hot.txt``、``user/hot-rule.txt``、``user/hot-server.txt``：存在時取代
  程式資料夾的同名檔案。
- ``user/LLM/*.py``：同名角色優先於程式資料夾 ``LLM/``，其餘角色照常載入。

優先順序（低 → 高）：程式資料夾預設值 → ``user/`` → Client 日常設定
（``%LOCALAPPDATA%\\CapsWriter\\client-settings.json``）→ Server 環境變數。
``CAPSWRITER_USER_DIR`` 可改用其他資料夾。本模組只用標準函式庫，Client-only
套件與 Docker image 都能 import。
"""

from __future__ import annotations

import ast
from contextlib import contextmanager
import filecmp
import importlib.util
import os
from pathlib import Path, PurePath
import shutil
import sys
from types import ModuleType
from typing import Iterable, Iterator, Mapping


USER_DIR_ENV = "CAPSWRITER_USER_DIR"
USER_DIR_NAME = "user"
ROLE_DIR_NAME = "LLM"
CONFIG_FILES = ("config_client.py", "config_server.py")
CLIENT_DATA_FILES = ("hot.txt", "hot-rule.txt")
SERVER_DATA_FILES = ("hot-server.txt",)
README_NAME = "README.txt"

TEMPLATES = {
    "config_client.py": "ClientConfig",
    "config_server.py": "ServerConfig",
}
TEMPLATE_EXAMPLES = {
    "config_client.py": (
        "#     addr = '192.168.1.20'",
        "#     paste_apps = ['WeiXin.exe', 'Telegram.exe', 'Code.exe']",
    ),
    "config_server.py": (
        "#     model_type = 'sensevoice'",
        "#     enable_tray = False",
    ),
}
README_TEXT = """\
CapsWriter 使用者設定資料夾
==========================

這個資料夾不在發行 ZIP 裡；把新版解壓覆蓋舊資料夾時，這裡的檔案不會被覆寫。
請在這裡修改設定，不要改程式資料夾裡的同名檔案。

  config_client.py   Client 進階設定：只寫要改的值（格式見檔內範例）
  config_server.py   Server 進階設定（完整版才有）
  hot.txt            熱詞
  hot-rule.txt       正則替換規則
  hot-server.txt     Server 熱詞（完整版）
  LLM/               LLM 角色；同名檔案優先於程式資料夾的 LLM/

優先順序（低 -> 高）：程式資料夾預設值 -> user/ -> 設定視窗的日常設定
（%LOCALAPPDATA%\\CapsWriter\\client-settings.json）-> Server 環境變數。

從舊版資料夾匯入設定：
  start_client.exe --import-config "D:\\舊的\\CapsWriter-Offline"

說明文件：docs/en/user-config.md、docs/zh-TW/user-config.md
"""

_applied: dict[str, tuple[str, ...]] = {}
_suppressed = 0


class UserConfigError(RuntimeError):
    """``user/`` 內的設定檔無法載入。"""


def _warn(message: str) -> None:
    # 設定檔在 logger 初始化之前就會載入，因此直接寫 stderr。
    stream = sys.stderr
    if stream is not None:
        print(f"[CapsWriter] {message}", file=stream)


def program_root() -> Path:
    """程式資料夾：凍結版為 EXE 所在目錄，原始碼為本檔所在目錄。"""

    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def user_dir() -> Path:
    explicit = os.environ.get(USER_DIR_ENV, "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    return program_root() / USER_DIR_NAME


def data_file(name: str) -> Path:
    """回傳 ``user/<name>``（存在時），否則程式資料夾的預設檔案。"""

    candidate = user_dir() / name
    if candidate.is_file():
        return candidate
    return program_root() / name


def role_dirs() -> list[Path]:
    """LLM 角色資料夾，依優先順序排列（user/LLM 在前）。"""

    dirs = [program_root() / ROLE_DIR_NAME]
    user_roles = user_dir() / ROLE_DIR_NAME
    if user_roles.is_dir():
        dirs.insert(0, user_roles)
    return dirs


def role_files() -> list[Path]:
    """每個角色檔名只取優先順序最高的一份，依檔名排序。"""

    selected: dict[str, Path] = {}
    for directory in role_dirs():
        if not directory.is_dir():
            continue
        for path in directory.glob("*.py"):
            if path.name != "__init__.py":
                selected.setdefault(path.name, path)
    return [selected[name] for name in sorted(selected)]


def extend_role_package(package_path: list[str]) -> None:
    """讓 ``import LLM.<角色>`` 先找 user/LLM（由 LLM/__init__.py 呼叫）。"""

    user_roles = user_dir() / ROLE_DIR_NAME
    if user_roles.is_dir() and str(user_roles) not in package_path:
        package_path.insert(0, str(user_roles))


@contextmanager
def _overlays_suppressed() -> Iterator[None]:
    global _suppressed
    _suppressed += 1
    try:
        yield
    finally:
        _suppressed -= 1


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise UserConfigError(f"無法載入 {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise UserConfigError(f"無法載入 {path}：{type(exc).__name__}: {exc}") from exc
    return module


def _class_settings(cls: type) -> dict[str, object]:
    return {
        name: value
        for name, value in vars(cls).items()
        if not name.startswith("__")
        and not callable(value)
        and not isinstance(value, (classmethod, staticmethod, property))
    }


def apply_overlay(
    namespace: dict,
    filename: str,
    data_paths: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """把 ``user/<filename>`` 的類別屬性套用到設定模組的同名類別。

    ``data_paths`` 把 ``"類別.屬性"`` 對應到 user/ 內的資料檔；該資料檔存在時，
    屬性改指向它（使用者設定檔明確指定的值仍然優先）。
    """

    if _suppressed:
        return ()
    applied: list[str] = []
    for target, name in (data_paths or {}).items():
        class_name, attribute = target.split(".", 1)
        cls = namespace.get(class_name)
        candidate = user_dir() / name
        if isinstance(cls, type) and candidate.is_file():
            setattr(cls, attribute, candidate)

    path = user_dir() / filename
    if path.is_file():
        module = _load_module(path, f"_capswriter_user_{Path(filename).stem}")
        for class_name, overrides in vars(module).items():
            if class_name.startswith("_") or not isinstance(overrides, type):
                continue
            target = namespace.get(class_name)
            if not isinstance(target, type):
                _warn(f"略過 {path} 中不存在的設定類別 {class_name}")
                continue
            for attribute, value in _class_settings(overrides).items():
                if not hasattr(target, attribute):
                    _warn(f"略過 {path} 中不存在的設定 {class_name}.{attribute}")
                    continue
                setattr(target, attribute, value)
                applied.append(f"{class_name}.{attribute}")
    _applied[filename] = tuple(applied)
    return _applied[filename]


def applied_overrides(filename: str) -> tuple[str, ...]:
    """目前行程已從 ``user/<filename>`` 套用的 ``類別.屬性``。"""

    return _applied.get(filename, ())


def _template(filename: str) -> str:
    class_name = TEMPLATES[filename]
    lines = [
        f"# CapsWriter 使用者設定：升級時不會被覆寫（程式資料夾的 {filename} 會）。",
        f"# 只寫要改的值，其餘沿用程式資料夾的 {filename}；",
        "# 類別與屬性名稱必須與它相同，不存在的名稱會在啟動時警告並略過。",
        "# 取消下列註解並修改即可：",
        "#",
        f"# class {class_name}:",
        *TEMPLATE_EXAMPLES[filename],
        "",
    ]
    return "\n".join(lines)


def _write_new(path: Path, text: str) -> bool:
    if path.exists():
        return False
    path.write_text(text, encoding="utf-8")
    return True


def _copy_new(source: Path, destination: Path) -> bool:
    if destination.exists() or not source.is_file():
        return False
    shutil.copy2(source, destination)
    return True


def seed(*, server: bool = False) -> list[Path]:
    """第一次啟動時建立 user/ 與範本；永遠不覆寫既有檔案。

    回傳新建立的路徑。資料夾無法寫入時只警告，程式繼續使用預設檔案。
    """

    root = program_root()
    target = user_dir()
    created: list[Path] = []
    filenames = ["config_client.py", *CLIENT_DATA_FILES]
    if server:
        filenames = ["config_server.py", *SERVER_DATA_FILES]
    try:
        target.mkdir(parents=True, exist_ok=True)
        if _write_new(target / README_NAME, README_TEXT):
            created.append(target / README_NAME)
        for name in filenames:
            destination = target / name
            if name in TEMPLATES:
                if not (root / name).is_file():
                    continue
                written = _write_new(destination, _template(name))
            else:
                written = _copy_new(root / name, destination)
            if written:
                created.append(destination)
        roles = target / ROLE_DIR_NAME
        if not server and not roles.exists() and (root / ROLE_DIR_NAME).is_dir():
            roles.mkdir()
            created.append(roles)
            for role in sorted((root / ROLE_DIR_NAME).glob("*.py")):
                if role.name != "__init__.py" and _copy_new(role, roles / role.name):
                    created.append(roles / role.name)
    except OSError as exc:
        _warn(f"無法建立使用者設定資料夾 {target}：{exc}；本次沿用程式資料夾的設定")
    return created


# ---- 從舊版資料夾匯入 ----

def _render_value(value: object) -> str | None:
    if isinstance(value, PurePath):
        return f"Path({value.as_posix()!r})"
    text = repr(value)
    try:
        if ast.literal_eval(text) == value:
            return text
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        pass
    return None


def _diff_config(old_file: Path, new_file: Path) -> tuple[list[str], list[str]]:
    """回傳 (新設定檔內容行, 無法轉換而略過的 類別.屬性)。"""

    with _overlays_suppressed():
        old = _load_module(old_file, "_capswriter_import_old")
        new = _load_module(new_file, "_capswriter_import_new")
    lines: list[str] = []
    skipped: list[str] = []
    needs_path = False
    for class_name, new_cls in vars(new).items():
        old_cls = getattr(old, class_name, None)
        if class_name.startswith("_") or not isinstance(new_cls, type) or not isinstance(old_cls, type):
            continue
        defaults = _class_settings(new_cls)
        changed: list[str] = []
        for attribute, value in _class_settings(old_cls).items():
            if attribute not in defaults or defaults[attribute] == value:
                continue
            rendered = _render_value(value)
            if rendered is None:
                skipped.append(f"{class_name}.{attribute}")
                continue
            needs_path = needs_path or rendered.startswith("Path(")
            changed.append(f"    {attribute} = {rendered}")
        if changed:
            lines += ["", f"class {class_name}:", *changed]
    header = [
        f"# 由 --import-config 從 {old_file} 產生；只列出與目前預設值不同的設定。",
    ]
    if needs_path:
        header.append("from pathlib import Path")
    return header + lines + [""], skipped


def _has_code(path: Path) -> bool:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return True
    return False


def _import_file(source: Path, destination: Path, default: Path) -> str:
    """複製舊檔；既有 user 檔若已被修改（不同於預設）則保留不動。"""

    if not source.is_file():
        return "missing"
    if default.is_file() and filecmp.cmp(source, default, shallow=False):
        return "unchanged"
    if destination.exists() and not (
        default.is_file() and filecmp.cmp(destination, default, shallow=False)
    ):
        if filecmp.cmp(source, destination, shallow=False):
            return "unchanged"
        return "kept"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return "imported"


def import_legacy(old_root: Path) -> list[str]:
    """把舊版程式資料夾內被修改過的設定匯入 user/，回傳逐項報告。

    設定檔只寫入與目前預設值不同的屬性；熱詞與角色檔只匯入與預設不同者。
    不覆寫 user/ 內已被修改的檔案。
    """

    old_root = Path(old_root).expanduser().resolve()
    root = program_root()
    if old_root == root:
        raise UserConfigError("請指定舊版的程式資料夾，而不是目前這一份")
    if not any((old_root / name).is_file() for name in CONFIG_FILES):
        raise UserConfigError(f"{old_root} 不是 CapsWriter 程式資料夾（找不到 config_client.py）")

    target = user_dir()
    target.mkdir(parents=True, exist_ok=True)
    _write_new(target / README_NAME, README_TEXT)
    report: list[str] = []

    for name in CONFIG_FILES:
        old_file, new_file, destination = old_root / name, root / name, target / name
        if not old_file.is_file() or not new_file.is_file():
            continue
        if destination.exists() and _has_code(destination):
            report.append(f"保留 {destination}（已有自訂內容，未覆寫）")
            continue
        lines, skipped = _diff_config(old_file, new_file)
        destination.write_text("\n".join(lines), encoding="utf-8")
        changed = sum(1 for line in lines if line.startswith("    "))
        report.append(f"匯入 {destination}：{changed} 項設定")
        for item in skipped:
            report.append(f"  未匯入 {item}：無法自動轉換，請手動設定")

    # 僅 Client 的安裝沒有 Server，不匯入 Server 熱詞。
    has_server = (root / "config_server.py").is_file()
    data_names: Iterable[str] = (*CLIENT_DATA_FILES, *(SERVER_DATA_FILES if has_server else ()))
    for name in data_names:
        status = _import_file(old_root / name, target / name, root / name)
        if status == "imported":
            report.append(f"匯入 {target / name}")
        elif status == "kept":
            report.append(f"保留 {target / name}（與舊版不同且已修改，請手動合併）")

    old_roles = old_root / ROLE_DIR_NAME
    if old_roles.is_dir():
        for role in sorted(old_roles.glob("*.py")):
            if role.name == "__init__.py":
                continue
            destination = target / ROLE_DIR_NAME / role.name
            status = _import_file(role, destination, root / ROLE_DIR_NAME / role.name)
            if status == "imported":
                report.append(f"匯入 {destination}")
            elif status == "kept":
                report.append(f"保留 {destination}（與舊版不同且已修改，請手動合併）")
    return report
