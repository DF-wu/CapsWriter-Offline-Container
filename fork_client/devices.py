"""PortAudio input identities without importing or initializing audio at import time."""
from __future__ import annotations

from collections import Counter


def valid_input_device(value) -> bool:
    def text(item):
        return isinstance(item, str) and bool(item.strip()) and len(item) <= 500

    if value is None or text(value):
        return True
    return (isinstance(value, dict)
            and set(value) in ({"name", "hostapi"}, {"name", "hostapi", "index"})
            and text(value["name"]) and text(value["hostapi"])
            and ("index" not in value or type(value["index"]) is int and value["index"] >= 0))


def device_label(selection) -> str:
    if not isinstance(selection, dict):
        return selection or ""
    label = f"{selection['name']} [{selection['hostapi']}]"
    if "index" in selection:
        label += f"（裝置 #{selection['index']}）"
    return label


def _inputs(sounddevice):
    hostapis = sounddevice.query_hostapis()
    return [
        (index, {"name": device["name"], "hostapi": hostapis[device["hostapi"]]["name"]})
        for index, device in enumerate(sounddevice.query_devices())
        if device["max_input_channels"] > 0
    ]


def input_device_choices(sounddevice) -> list[dict]:
    devices = _inputs(sounddevice)
    counts = Counter((value["name"], value["hostapi"]) for _, value in devices)
    choices = []
    for index, value in devices:
        # Only identical names within one API require a volatile PortAudio index.
        if counts[value["name"], value["hostapi"]] > 1:
            value = {**value, "index": index}
        choices.append({"label": device_label(value), "value": value})
    return choices


def resolve_input_device(selection, sounddevice):
    """Resolve persisted identity to today's index; retain legacy string semantics."""
    if not isinstance(selection, dict):
        return selection
    if not valid_input_device(selection):
        raise ValueError("麥克風設定格式無效；請重新選擇裝置")
    matches = [index for index, value in _inputs(sounddevice)
               if value["name"] == selection["name"] and value["hostapi"] == selection["hostapi"]]
    if "index" in selection:
        if selection["index"] in matches:
            return selection["index"]
        raise ValueError("麥克風裝置編號已變更或裝置不存在；請重新整理並選擇裝置")
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError("找不到指定名稱與音訊介面的麥克風；請重新選擇裝置")
    raise ValueError("同一音訊介面有多個同名麥克風；請重新整理並選擇裝置編號")
