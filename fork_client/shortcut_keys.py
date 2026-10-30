"""Dependency-free shortcut syntax shared by settings and desktop listeners.

Chords are sorted lexically, use '+' as the separator and 'plus' for a literal
plus. vk_N denotes a platform-specific virtual key (decimal 1..65535).
Generic ctrl, alt and shift match either side. Suffixes _l/_r are
side-specific; left_ctrl/left_alt are aliases for ctrl_l/alt_l.
Mouse side buttons x1/x2 are accepted only as singletons.
"""
from __future__ import annotations

import re

KEYBOARD_KEYS = frozenset({
    'ctrl', 'alt', 'ctrl_l', 'ctrl_r', 'shift', 'shift_l', 'shift_r',
    'alt_l', 'alt_r', 'alt_gr', 'cmd', 'cmd_l', 'cmd_r',
    'space', 'enter', 'tab', 'backspace', 'delete', 'insert', 'home', 'end',
    'page_up', 'page_down', 'esc', 'caps_lock', 'num_lock', 'scroll_lock',
    'print_screen', 'pause', 'menu', 'up', 'down', 'left', 'right', 'plus',
    'media_play_pause', 'media_volume_mute', 'media_volume_down',
    'media_volume_up', 'media_previous', 'media_next',
    'numpad_add', 'numpad_subtract', 'numpad_multiply', 'numpad_divide',
    'numpad_decimal', 'numpad_separator',
} | {f'f{i}' for i in range(1, 25)} | {f'numpad{i}' for i in range(10)})

_ALIASES = {
    'control': 'ctrl', 'control_l': 'ctrl_l', 'control_r': 'ctrl_r',
    'left_ctrl': 'ctrl_l', 'left_alt': 'alt_l',
    'capslock': 'caps_lock', 'caps lock': 'caps_lock',
    'numlock': 'num_lock', 'scrolllock': 'scroll_lock',
    'escape': 'esc', 'return': 'enter', 'spacebar': 'space',
    'pageup': 'page_up', 'pagedown': 'page_down', 'del': 'delete',
    'left ctrl': 'ctrl_l', 'left control': 'ctrl_l',
    'right ctrl': 'ctrl_r', 'right control': 'ctrl_r',
    'left alt': 'alt_l', 'right alt': 'alt_r',
    'left shift': 'shift_l', 'right shift': 'shift_r',
    'altgr': 'alt_gr',
}


def _token(value: str) -> str:
    value = value.strip().lower()
    if value.startswith('<') and value.endswith('>') and len(value) > 2:
        value = value[1:-1].strip()
    else:
        # A literal '<' or '>' is also a valid character key.
        if len(value) > 1 and ('<' in value or '>' in value):
            raise ValueError('Malformed angle brackets in shortcut')
    value = _ALIASES.get(value, value)
    if value in KEYBOARD_KEYS or value in {'x1', 'x2'}:
        return value
    if re.fullmatch(r'vk_[0-9]+', value):
        number = int(value[3:])
        if 1 <= number <= 65535:
            return f'vk_{number}'
    if len(value) == 1 and value.isprintable() and not value.isspace() and value != '+':
        return value
    raise ValueError(f'Invalid shortcut key: {value!r}')


def parse_shortcut(value: str) -> tuple[str, ...]:
    """Return canonical sorted tokens; reject empty, duplicate or mouse chords."""
    if value == ' ':
        return ('space',)
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Shortcut must be a nonempty string')
    if value.strip() == '+':
        return ('plus',)
    if '<+>' in value:
        raise ValueError("Use the named key 'plus' for a literal plus")
    tokens = tuple(_token(part) for part in value.split('+'))
    if len(set(tokens)) != len(tokens):
        raise ValueError('Repeated shortcut key (including aliases)')
    if len(tokens) > 1 and {'x1', 'x2'}.intersection(tokens):
        raise ValueError('Mouse buttons cannot be combined with keyboard keys')
    return tuple(sorted(tokens))


def normalize_shortcut(value: str) -> str:
    """Return a deterministic canonical shortcut, or raise ValueError."""
    return '+'.join(parse_shortcut(value))


def event_key_name(name: str) -> str:
    """Normalize an event name without collapsing physical modifier sides."""
    if len(name) == 1 and 1 <= ord(name) <= 26:
        return chr(ord(name) + 96)  # Ctrl+A..Z can arrive as ASCII control codes.
    if name == ' ':
        return 'space'
    if name == '+':
        return 'plus'
    name = name.lower()
    return {'ctrl': 'ctrl_l', 'alt': 'alt_l', 'shift': 'shift_l', 'cmd': 'cmd_l'}.get(name, name)


def key_matches(token: str, names: frozenset[str]) -> bool:
    """Match a canonical token against names for one physical key."""
    alternatives = {
        'ctrl': {'ctrl_l', 'ctrl_r'},
        'alt': {'alt_l', 'alt_r', 'alt_gr'},
        'shift': {'shift_l', 'shift_r'},
        'alt_gr': {'alt_gr', 'alt_r'}, 'alt_r': {'alt_gr', 'alt_r'},
        'cmd': {'cmd', 'cmd_l', 'cmd_r'},
    }
    return bool(alternatives.get(token, {token}).intersection(names))
