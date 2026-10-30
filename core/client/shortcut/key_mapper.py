# coding: utf-8
"""
按键映射相关

处理按键名称和虚拟键码之间的转换，以及相关常量定义
"""

from platform import system

from fork_client.shortcut_keys import event_key_name

from . import logger


_key_translator = None
_SPECIAL_KEYS = None


def _get_keyboard_module():
    from pynput import keyboard

    return keyboard


def _get_key_translator():
    """Load the private Win32 translator only on Windows."""

    global _key_translator
    if system() != 'Windows':
        return None
    if _key_translator is None:
        from pynput._util.win32 import KeyTranslator

        _key_translator = KeyTranslator()
    return _key_translator


def _get_special_keys():
    global _SPECIAL_KEYS
    if _SPECIAL_KEYS is None:
        keyboard = _get_keyboard_module()
        _SPECIAL_KEYS = {
            key.value.vk: key
            for key in keyboard.Key
        }
    return _SPECIAL_KEYS

# 小键盘按键映射（VK -> 名称）
NUMPAD_KEYS = {
    0x60: 'numpad0',  0x61: 'numpad1',  0x62: 'numpad2',  0x63: 'numpad3',
    0x64: 'numpad4',  0x65: 'numpad5',  0x66: 'numpad6',  0x67: 'numpad7',
    0x68: 'numpad8',  0x69: 'numpad9',
    0x6A: 'numpad_multiply',  # *
    0x6B: 'numpad_add',       # +
    0x6C: 'numpad_separator', # （通常未使用）
    0x6D: 'numpad_subtract',  # -
    0x6E: 'numpad_decimal',   # 小数点
    0x6F: 'numpad_divide',    # /
}

# Windows 键盘消息常量
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
WM_SYSKEYDOWN = 0x0104
WM_SYSKEYUP = 0x0105

# Windows 鼠标消息常量
WM_XBUTTONDOWN = 0x020B
WM_XBUTTONUP = 0x020C
XBUTTON1 = 0x0001
XBUTTON2 = 0x0002

# 按键消息集合
KEYBOARD_MESSAGES = (WM_KEYDOWN, WM_KEYUP, WM_SYSKEYDOWN, WM_SYSKEYUP)
KEY_UP_MESSAGES = (WM_KEYUP, WM_SYSKEYUP)
KEY_DOWN_MESSAGES = (WM_KEYDOWN, WM_SYSKEYDOWN)
MOUSE_MESSAGES = (WM_XBUTTONDOWN, WM_XBUTTONUP)

# 可恢复的切换键（需要录音完成后恢复状态的锁键）
RESTORABLE_KEYS = {
    'caps_lock',    # 大写锁定
    'num_lock',     # 数字键盘锁定
    'scroll_lock',  # 滚动锁定
}


class KeyMapper:
    """按键映射器"""

    # pynput 特殊键对象缓存
    _SPECIAL_KEY_OBJECTS = None

    @staticmethod
    def scan_to_name(scan_code: int) -> str:
        """Get an unmodified character from pynput's cached active layout."""
        try:
            translator = _get_key_translator()
            char = translator.char_from_scan(scan_code) if translator is not None else None
            return event_key_name(char) if char else ''
        except (IndexError, KeyError, OSError, ValueError):
            return ''

    @classmethod
    def _get_special_key_objects(cls):
        """获取 pynput 特殊键对象（延迟初始化）"""
        if cls._SPECIAL_KEY_OBJECTS is None:
            keyboard = _get_keyboard_module()
            # 直接从 pynput keyboard.Key 枚举构建 name -> key 映射，
            # 覆盖全部特殊键（含 insert/home/end/方向键/page_up 等），避免手工枚举遗漏
            cls._SPECIAL_KEY_OBJECTS = {
                key.name: key
                for key in keyboard.Key
            }
        return cls._SPECIAL_KEY_OBJECTS

    @staticmethod
    def vk_to_name(vk: int) -> str:
        """
        将虚拟键码转换为按键名称

        Args:
            vk: 虚拟键码

        Returns:
            str: 按键名称（与 Shortcut.key 格式一致）
        """
        # Function keys beyond F12 are absent from some pynput Key enums.
        if 0x70 <= vk <= 0x87:
            return f'f{vk - 0x6F}'
        physical = {
            0x10: 'shift_l', 0x11: 'ctrl_l', 0x12: 'alt_l',
            0xA0: 'shift_l', 0xA1: 'shift_r',
            0xA2: 'ctrl_l', 0xA3: 'ctrl_r',
            0xA4: 'alt_l', 0xA5: 'alt_r',
        }
        if vk in physical:
            return physical[vk]
        # 首先检查是否是特殊键（pynput Key 枚举）
        special_keys = _get_special_keys()
        if vk in special_keys:
            return event_key_name(special_keys[vk].name)

        # 检查是否是小键盘按键
        if vk in NUMPAD_KEYS:
            return NUMPAD_KEYS[vk]

        # 使用 pynput 的 KeyTranslator 获取字符（字母、数字、符号键）
        try:
            translator = _get_key_translator()
            if translator is not None:
                params = translator(vk, is_press=True)
                if 'char' in params and params['char'] is not None:
                    return event_key_name(params['char'])
        except Exception:
            pass

        if 0x30 <= vk <= 0x39 or 0x41 <= vk <= 0x5A:
            return chr(vk).lower()
        # 未知键码，返回 vk_ 格式
        return f'vk_{vk}'

    @staticmethod
    def name_to_key(key_name: str):
        """
        将按键名称转换为 pynput 按键对象

        Args:
            key_name: 按键名称

        Returns:
            pynput 按键对象或 None
        """
        keyboard = _get_keyboard_module()

        key_name = {'left_ctrl': 'ctrl_l', 'left_alt': 'alt_l'}.get(key_name, key_name)

        # 特殊按键
        special_keys = KeyMapper._get_special_key_objects()
        if key_name in special_keys:
            return special_keys[key_name]
        alias = {'shift_l': 'shift', 'cmd_l': 'cmd', 'alt_r': 'alt_gr',
                 'ctrl_l': 'ctrl', 'alt_l': 'alt'}.get(key_name)
        if alias in special_keys:
            return special_keys[alias]

        if key_name == 'plus':
            return keyboard.KeyCode.from_char('+')
        if key_name.startswith('vk_'):
            return keyboard.KeyCode.from_vk(int(key_name[3:]))
        for vk, name in NUMPAD_KEYS.items():
            if name == key_name:
                if system() == 'Windows':
                    return keyboard.KeyCode.from_vk(vk)
                if system() == 'Linux':
                    x11_vk = 0xFFB0 + vk - 0x60 if vk <= 0x69 else 0xFFAA + vk - 0x6A
                    return keyboard.KeyCode.from_vk(x11_vk)
                return None
        if key_name.startswith('f') and key_name[1:].isdigit():
            number = int(key_name[1:])
            if 1 <= number <= 24:
                if system() == 'Windows':
                    return keyboard.KeyCode.from_vk(0x6F + number)
                if system() == 'Linux':
                    return keyboard.KeyCode.from_vk(0xFFBD + number)

        # 单个字符按键
        if len(key_name) == 1:
            return keyboard.KeyCode.from_char(key_name)

        logger.warning(f"未知按键名称: {key_name}")
        return None
