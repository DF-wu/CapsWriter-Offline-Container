#!/usr/bin/env python3
"""Rebuild publication SVGs and PNGs from compact observations, offline.

SVG uses explicit vector primitives; librsvg supplies PNG rasterization.
No inference, network, JavaScript, or third-party Python packages.
"""
from html import escape
import json
from pathlib import Path
from statistics import median
import subprocess

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIG = ROOT / "figures"
COLORS = {"cpu": "#586777", "hybrid": "#007f83", "migraphx": "#b95524"}
NAMES = {"cpu": "CPU", "hybrid": "CPU + Vulkan", "migraphx": "MIGraphX + Vulkan"}
ORDER = ["short", "reference", "mandarin", "long", "mandarin_long", "noninteger"]
LABELS = ["Short English", "Reference English", "Mandarin", "Long English", "Synthetic Mandarin-long", "Noninteger English"]


class SVG:
    def __init__(self, title, subtitle, height=680):
        self.height = height
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="{height}" viewBox="0 0 1120 {height}" role="img"><title>{escape(title)}</title><desc>{escape(subtitle)}</desc>',
                      '<rect width="1120" height="100%" fill="white"/>',
                      '<style>text{font-family:DejaVu Sans,Arial,sans-serif;fill:#20323f;font-size:15px}.small{font-size:13px;fill:#52616b}.title{font-size:25px;font-weight:700}.sub{font-size:15px;fill:#52616b}</style>']
        self.text(34, 40, title, cls="title")
        self.text(34, 68, subtitle, cls="sub")

    def text(self, x, y, text, cls="", anchor="start", fill=None):
        color = f' style="fill:{fill}"' if fill else ""
        self.parts.append(f'<text x="{x:.2f}" y="{y:.2f}" class="{cls}" text-anchor="{anchor}"{color}>{escape(str(text))}</text>')

    def rect(self, x, y, width, height, fill, stroke="none", radius=0):
        self.parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" rx="{radius}" fill="{fill}" stroke="{stroke}"/>')

    def line(self, x1, y1, x2, y2, stroke="#d9e0e4", dash=None, width=1):
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke}" stroke-width="{width}"{extra}/>')

    def dot(self, x, y, color, radius=4, fill="white"):
        self.parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{fill}" stroke="{color}" stroke-width="2"/>')

    def save(self, name):
        self.parts.append("</svg>")
        target = FIG / (name + ".svg")
        target.write_text("\n".join(self.parts) + "\n")
        subprocess.run(["rsvg-convert", "--width", "2240", "--output", str(FIG / (name + ".png")), str(target)], check=True, timeout=30)

    def key(self, y=100):
        for i, mode in enumerate(COLORS):
            self.rect(34 + i * 320, y - 12, 18, 13, COLORS[mode])
            self.text(62 + i * 320, y, NAMES[mode])


def load(name):
    return json.loads((DATA / name).read_text())


def warm(requests):
    s = SVG("Installed system: hybrid has the lowest repeat HTTP latency",
            "Bars: medians; outlined points: all repeats. n=3 reference/short, n=2 otherwise. Serial instrumented runs.", 750)
    s.key()
    left, right, top = 260, 1020, 145
    for tick in range(9):
        x = left + tick / 8 * (right - left)
        s.line(x, top - 5, x, 650)
        s.text(x, 680, tick, anchor="middle")
    for i, (name, label) in enumerate(zip(ORDER, LABELS)):
        y = top + i * 84
        duration = next(r["audio_s"] for r in requests if r["name"] == name)
        s.text(left - 15, y + 17, label, anchor="end")
        s.text(left - 15, y + 38, f"{duration:g} s audio", cls="small", anchor="end")
        for j, mode in enumerate(COLORS):
            values = [r["seconds"] for r in requests if r["name"] == name and r["mode"] == mode and r["phase"] == "repeat"]
            value = median(values)
            yy = y + j * 21
            width = value / 8 * (right - left)
            s.rect(left, yy, width, 16, COLORS[mode])
            for k, v in enumerate(values):
                s.dot(left + v / 8 * (right - left), yy + 8 + (k - (len(values) - 1) / 2) * 3, COLORS[mode], 3)
            s.text(left + max(values) / 8 * (right - left) + 10, yy + 13, f"{value:.3f}", cls="small")
    s.text((left + right) / 2, 717, "HTTP CLI file-to-transcript wall time (seconds); zero baseline", anchor="middle")
    s.save("http-repeat")


def sequence(requests):
    s = SVG("Changing or returning input shapes triggers large MIGraphX stalls",
            "Every HTTP request, original serial order (22 per mode). Connected points show order, not independent trials.", 730)
    s.key()
    left, right, top, bottom = 80, 1070, 145, 555
    for tick in [0, 5, 10, 15, 20, 25, 30]:
        y = bottom - tick / 30 * (bottom - top)
        s.line(left, y, right, y)
        s.text(left - 12, y + 5, tick, anchor="end")
    rows = sorted((r for r in requests if r["mode"] == "migraphx"), key=lambda r: r["index"])
    for r in rows:
        x = left + r["index"] / 21 * (right - left)
        if r["phase"] != "repeat":
            s.line(x, top, x, bottom, "#ccd3d6", "3 4")
        s.text(x, 581, r["index"] + 1, anchor="middle", cls="small")
        short = {"reference": "ref", "short": "short", "long": "long", "mandarin": "zh", "mandarin_long": "zh-long", "noninteger": "nonint"}[r["name"]]
        s.parts.append(f'<text transform="translate({x:.2f},604) rotate(55)" class="small">{short}</text>')
        if r["phase"] != "repeat":
            s.text(x, bottom - r["seconds"] / 30 * (bottom - top) - 12, f'{r["seconds"]:.2f}', anchor="middle", cls="small")
    for mode, color in COLORS.items():
        values = sorted((r for r in requests if r["mode"] == mode), key=lambda r: r["index"])
        points = [(left + r["index"] / 21 * (right - left), bottom - r["seconds"] / 30 * (bottom - top)) for r in values]
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            s.line(x1, y1, x2, y2, color, width=1.6)
        for x, y in points:
            s.dot(x, y, color, 4)
    s.text(80, 688, "Y: HTTP wall seconds. X: request number / fixture. Dashed guides: first, new, or returned shapes.")
    s.text(80, 712, "Installed timings have no isolated compiler timer; stalls are consistent with compilation, not measured compiler durations.", cls="small")
    s.save("http-sequence")


def numerical():
    rows = load("numerical.json")
    s = SVG("All 22 installed MIGraphX embedding comparisons fail the original gate",
            "Acceptance requires relative L2 <2% AND cosine >0.999. Text/token agreement does not change this criterion.", 650)
    for panel, key, title, low, high, gate in [(0, "embedding_relative_l2", "CPU-relative L2 (%)", 0, 6.5, 2),
                                               (1, "embedding_cosine", "Cosine similarity (expanded scale)", .998, 1.0001, .999)]:
        left = 270 + panel * 440
        width = 340
        def x(v):
            return left + (v - low) / (high - low) * width
        ticks = [0, 2, 4, 6] if panel == 0 else [.998, .999, 1.0]
        for t in ticks:
            s.line(x(t), 150, x(t), 510)
            s.text(x(t), 542, f"{t:g}", anchor="middle")
        s.line(x(gate), 135, x(gate), 510, "#8a2942", "5 4", 2)
        s.text(x(gate), 122, "strict gate", anchor="middle", fill="#8a2942")
        s.text(left + width / 2, 580, title, anchor="middle")
        for i, (name, label) in enumerate(zip(ORDER, LABELS)):
            group = [r for r in rows if r["name"] == name]
            y = 172 + i * 59
            if panel == 0:
                s.text(245, y + 5, f"{label} (n={len(group)})", anchor="end")
            value = group[0][key] * (100 if panel == 0 else 1)
            for j, r in enumerate(group):
                v = r[key] * (100 if panel == 0 else 1)
                s.dot(x(v), y + (j - (len(group) - 1) / 2) * 6, COLORS["migraphx"], 3)
            s.text(x(value) + 9, y + 5, f"{value:.3f}" if panel == 0 else f"{value:.7f}", cls="small")
    s.text(34, 625, "Each point is a retained request comparison; overlapping x values are vertically separated. L2 uses zero baseline; cosine axis is expanded.", cls="small")
    s.save("numerical-gate")


def components():
    s = SVG("Readiness and the first reference request expose different costs",
            "Installed system; one observed startup / first reference per mode. Nested components are shown separately, never stacked.", 900)
    s.key()
    panels = [("Startup", [("readiness", "Process start → ready"), ("engine_load", "Engine construction"),
                           ("encoder_load_and_builtin_warmup", "Encoder + 2 s warmup"), ("decoder_model_load", "Decoder model load")], 32, 145),
              ("First reference request", [("pipeline_s", "Worker pipeline"), ("encoder_s", "Encoder total"),
                                           ("backend_s", "ONNX backend"), ("decoder_s", "Decoder total")], 24, 515)]
    loads, tasks = load("loads.json"), load("tasks.json")
    for title, metrics, maximum, top in panels:
        s.text(34, top - 10, title)
        left, width = 275, 730
        for tick in range(0, maximum + 1, 8 if maximum == 32 else 6):
            x = left + tick / maximum * width
            s.line(x, top + 5, x, top + 292)
            s.text(x, top + 315, tick, anchor="middle", cls="small")
        for i, (key, label) in enumerate(metrics):
            y = top + 12 + i * 72
            s.text(255, y + 25, label, anchor="end")
            for j, mode in enumerate(COLORS):
                value = next(r["seconds"] for r in loads if r["mode"] == mode and r["stage"] == key) if title == "Startup" else next(r[key] for r in tasks if r["mode"] == mode and r["sequence"] == 0)
                bar = value / maximum * width
                s.rect(left, y + j * 19, bar, 14, COLORS[mode])
                s.text(left + bar + 8, y + j * 19 + 12, f"{value:.3f}", cls="small")
        s.text(640, top + 342, "Host wall time (seconds); zero baseline", anchor="middle", cls="small")
    s.save("startup-components")


def pipeline():
    s = SVG("One recognition pipeline, three measured placements",
            "Schematic: boxes encode responsibility and placement, not measured time or proportional compute.", 660)
    labels = [(35, 190, "Client / transport", "HTTP file or PCM WS"), (265, 190, "CPU preparation", "FFmpeg → PCM → mel"),
              (495, 190, "FP16 split encoder", "ONNX frontend + backend"), (735, 190, "Q5_K_M decoder", "llama.cpp b7798"), (965, 190, "Text", "Format")]
    for x, y, title, detail in labels:
        width = 190 if x < 965 else 120
        s.rect(x, y, width, 82, "#eef3f5", "#bac7ce", 8)
        s.text(x + width / 2, y + 30, title, anchor="middle")
        s.text(x + width / 2, y + 56, detail, anchor="middle", cls="small")
        if x < 965:
            end = x + width
            s.line(end + 3, y + 41, end + 30, y + 41, "#52616b", width=2)
            s.text(end + 25, y + 47, "›")
    s.text(590, 165, "[T, 2048] audio embeddings", anchor="middle", cls="small")
    s.text(829, 165, "Prompt + autoregressive tokens", anchor="middle", cls="small")
    for j, mode in enumerate(COLORS):
        y = 325 + j * 80
        s.text(35, y + 25, NAMES[mode], fill=COLORS[mode])
        for x, width, text in [(265, 190, "CPU"), (495, 190, "MIGraphX / RX570" if mode == "migraphx" else "ORT / CPU"),
                               (735, 190, "CPU" if mode == "cpu" else "Vulkan / RX570")]:
            s.rect(x, y, width, 46, "#f8fafb", COLORS[mode], 6)
            s.text(x + width / 2, y + 29, text, anchor="middle")
    s.text(35, 604, "Hybrid: all 29 reported decoder layers placed on Vulkan. MIGraphX: both encoder graphs execute strictly on GPU.")
    s.text(35, 630, "CPU remains responsible for mel, transport, queue and formatting. Optional aligner disabled; no CUDA execution on RX570.", cls="small")
    s.save("pipeline")


def historical():
    rows = load("historical.json")["pairs"]
    s = SVG("Nov10 paired speech study: no established overall Q6 quality gain",
            "Tesla P4 study, separate from RX570. Point: Q6 minus Q5; line: retained paired-bootstrap 95% interval.", 550)
    left, width, low, high = 330, 700, -.3, .9
    def x(v):
        return left + (v - low) / (high - low) * width
    for tick in [-.2, 0, .2, .4, .6, .8]:
        s.line(x(tick), 125, x(tick), 380, "#87979f" if tick == 0 else "#d9e0e4", "5 4" if tick == 0 else None)
        s.text(x(tick), 411, f"{tick:+.1f}", anchor="middle")
    for i, (lang, mode, label) in enumerate([("zh", "normalized", "Mandarin normalized CER"), ("en", "normalized", "English normalized WER"),
                                            ("zh", "raw", "Mandarin raw CER"), ("en", "raw", "English raw WER")]):
        row = next(r for r in rows if r["phase"] == "quality" and r["language"] == lang and r["source"] is None and r["mode"] == mode)
        y = 160 + i * 65
        lo, hi = row["bootstrap95_pp"]
        s.text(left - 20, y + 5, f'{label} (n={row["n"]})', anchor="end")
        s.line(x(lo), y, x(hi), y, COLORS["hybrid"], width=3)
        for v in [lo, hi]:
            s.line(x(v), y - 7, x(v), y + 7, COLORS["hybrid"], width=2)
        s.dot(x(row["difference_pp"]), y, COLORS["hybrid"], 5, COLORS["hybrid"])
        s.text(x(hi) + 10, y + 5, f'{row["difference_pp"]:+.3f} pp', cls="small")
    s.text(680, 452, "Error-rate difference (percentage points); lower favors Q6", anchor="middle")
    s.text(34, 496, "10,000 paired utterance resamples, seed 1729. Intervals are retained original analysis, not regenerated by the stdlib validator.", cls="small")
    s.text(34, 522, "176 valid paired recordings; incomplete acquisition and native failures remain. No such intervals are inferred for installed n=2/3 repeats.", cls="small")
    s.save("q5-q6-quality")


def main():
    FIG.mkdir(exist_ok=True)
    requests = load("requests.json")
    warm(requests)
    sequence(requests)
    numerical()
    components()
    pipeline()
    historical()
    print("Generated six SVG / PNG figure pairs from compact observations.")


if __name__ == "__main__":
    main()
