#!/usr/bin/env python3
"""Build a self-contained English HTML reading/print edition, offline.

Requires the already available Python-Markdown package and DejaVu fonts.
PDF is optional: --pdf uses an existing Playwright Chromium installation.
It never starts a server, downloads a browser, or accesses the network.
"""
import argparse
import base64
import hashlib
from html import escape
from html.parser import HTMLParser
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import quote, urlsplit

import markdown

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
BASE_URL = "https://github.com/DF-wu/CapsWriter-Offline-Container/blob/docs/rx570-research-20261115/"


class TableNumbers(HTMLParser):
    """Keep short numeric tokens intact, with deliberate statistic separators."""
    number = re.compile(r"(?<![\w.])[-+−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w.])")

    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.parts, self.tables, self.code = [], 0, 0

    def handle_starttag(self, tag, attrs):
        self.tables += tag == "table"
        self.code += tag == "code"
        self.parts.append(self.get_starttag_text())

    def handle_startendtag(self, tag, attrs):
        self.parts.append(self.get_starttag_text())

    def handle_endtag(self, tag):
        self.tables -= tag == "table"
        self.code -= tag == "code"
        self.parts.append(f"</{tag}>")

    def handle_data(self, text):
        text = escape(text, quote=False)
        if self.tables and not (self.code and re.search(r"[A-Za-z_]", text)):
            text = self.number.sub(lambda m: f'<span class="number">{m[0]}</span>' if len(m[0]) <= 18 else m[0], text)
            text = re.sub(r'</span>\s*/\s*(?=<span class="number">)', '</span>/<br>', text)
            text = re.sub(r'</span>,\s*(?=<span class="number">)', '</span>,<wbr>', text)
            text = text.replace(" [", "<br>[")
        self.parts.append(text)

    def handle_entityref(self, name):
        self.parts.append(f"&{name};")

    def handle_charref(self, name):
        self.parts.append(f"&#{name};")


def data_url(path, mime):
    return "data:" + mime + ";base64," + base64.b64encode(path.read_bytes()).decode()


def render_document(path):
    content = markdown.markdown(path.read_text(), extensions=["tables", "fenced_code", "toc"])
    def image(match):
        target = (path.parent / match.group(1)).resolve()
        if target.suffix == ".png":
            target = target.with_suffix(".svg")
        mime = "image/svg+xml" if target.suffix == ".svg" else "image/png"
        return 'src="' + data_url(target, mime) + '"'
    content = re.sub(r'src="([^":]+\.(?:png|svg))"', image, content)
    content = re.sub(r'<p>(<img[^>]+>)</p>\s*<p>(<strong>Figure \d+\..*?)</p>',
                     r'<figure>\1<figcaption>\2</figcaption></figure>', content, flags=re.DOTALL)
    def link(match):
        target = match.group(1)
        if urlsplit(target).scheme or target.startswith("#"):
            return match.group(0)
        clean = target.split("#", 1)[0]
        if clean.endswith("source-history.md"):
            return 'href="#appendix-a"'
        if clean.endswith("live-history.md"):
            return 'href="#appendix-b"'
        resolved = (path.parent / clean).resolve()
        relative = resolved.relative_to(REPO)
        return 'href="' + BASE_URL + quote(str(relative)) + '"'
    content = re.sub(r'href="([^"]+)"', link, content)
    numbers = TableNumbers()
    numbers.feed(content)
    return "".join(numbers.parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", action="store_true")
    args = parser.parse_args()
    font_records = []
    faces = []
    for family, query, weight in [("Reading", "DejaVu Serif", "400"), ("Labels", "DejaVu Sans", "400"),
                                   ("Labels", "DejaVu Sans:style=Bold", "700")]:
        font = Path(subprocess.check_output(["fc-match", "--format=%{file}", query], text=True, timeout=10))
        font_records.append({"family": family, "file": font.name, "sha256": hashlib.sha256(font.read_bytes()).hexdigest()})
        faces.append(f"@font-face{{font-family:{family};font-weight:{weight};src:url('{data_url(font, 'font/ttf')}')}}")
    style = """*{box-sizing:border-box}html{scroll-behavior:auto}body{margin:0;background:#edf1f3;color:#20323f;font:17px/1.66 Reading,serif;overflow-wrap:anywhere}
main{max-width:1120px;margin:40px auto;background:white;padding:64px 72px;box-shadow:0 8px 34px #20323f12}
h1,h2,h3,h4,nav,table,figcaption,.eyebrow{font-family:Labels,sans-serif}h1{font-size:38px;line-height:1.18;letter-spacing:-.7px;margin:12px 0 28px}
h2{font-size:27px;line-height:1.3;margin-top:2.1em;padding-top:14px;border-top:2px solid #dbe5e8}h3{font-size:20px;margin-top:1.7em}h4{font-size:17px}
.eyebrow{font-size:13px;text-transform:uppercase;letter-spacing:2px;color:#007f83;font-weight:700}.edition{border-top:5px solid #007f83;padding-top:22px}
a{color:#006e75;text-decoration-thickness:1px;text-underline-offset:3px;overflow-wrap:anywhere}p{margin:1em 0}strong{font-family:Labels,sans-serif;font-weight:700}
blockquote{margin:22px 0;border-left:3px solid #9bbec0;padding:1px 20px;font-size:14px;color:#52616b}img{width:100%;height:auto;display:block;margin:28px auto 12px}figure{margin:24px 0;break-inside:avoid}figcaption{font-size:13px;line-height:1.5;color:#52616b}
table{border-collapse:collapse;width:100%;font-size:12px;line-height:1.5;margin:24px 0;table-layout:auto}th{background:#eaf0f2;text-align:left;color:#20323f}
th,td{padding:9px 8px;border-bottom:1px solid #d9e0e4;vertical-align:top;overflow-wrap:normal;word-break:normal}tr:nth-child(even) td{background:#f8fafb}
.number{display:inline-block;white-space:nowrap;overflow-wrap:normal;word-break:normal;hyphens:none;font-variant-numeric:tabular-nums}
code{font:12px/1.5 DejaVu Sans Mono,monospace;overflow-wrap:anywhere;white-space:normal}pre{padding:18px;background:#f3f6f7;overflow-wrap:anywhere;white-space:pre-wrap}
pre code{white-space:pre-wrap}ul,ol{padding-left:1.5em}li{margin:.45em 0}.appendix{margin-top:80px;padding-top:25px;border-top:5px solid #586777}
nav{font-size:14px;display:flex;gap:20px;flex-wrap:wrap;margin-bottom:30px}.note{font:14px/1.5 Labels,sans-serif;color:#52616b}
@media screen and (max-width:750px){body{font-size:16px}main{padding:28px 18px;margin:0;box-shadow:none}h1{font-size:29px}h2{font-size:23px}table{font-size:10px;display:block;overflow-x:auto}th,td{padding:6px 4px}}
@page{size:A4;margin:18mm 15mm 20mm} @media print{body{background:white;font-size:10pt;line-height:1.5}main{padding:0;margin:0;max-width:none;box-shadow:none}
h1{font-size:26pt}h2{font-size:17pt;margin-top:1.5em}h3{font-size:13pt}h1,h2,h3,h4{break-after:avoid}p,li{orphans:3;widows:3}table{font-size:7.5pt}
tr,img,pre{break-inside:avoid}thead{display:table-header-group}img{max-height:238mm;object-fit:contain}.appendix{break-before:page;margin-top:0}nav{display:none}a{color:#20323f;text-decoration:none}}
""".replace("\\+", "")
    main_path = ROOT.parent / "rx570-qwen3-asr-research-20261115.md"
    body = '<main><div class="edition"><div class="eyebrow">CapsWriter · Research archive · 15 November 2026</div>'
    body += '<nav><a href="#abstract">Abstract</a><a href="#6-installed-system-results">Installed results</a><a href="#appendix-a">Quantization / sources</a><a href="#appendix-b">Live audit</a></nav>'
    body += render_document(main_path) + '</div>'
    for anchor, path in [("appendix-a", ROOT / "source-history.md"), ("appendix-b", ROOT / "live-history.md")]:
        body += f'<section class="appendix" id="{anchor}">' + render_document(path) + '</section>'
    body += '<p class="note">End of retained-evidence report. Figures and DejaVu fonts are embedded; this document makes no network requests.</p>'
    body += '<details class="font-notices"><summary>Embedded font redistribution notices</summary><pre>' + escape((ROOT / "FONT-LICENSES.txt").read_text()) + '</pre></details></main>'
    target = ROOT / "report.html"
    target.write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light"><title>Qwen3-ASR / RX570 research archive</title><style>' + "".join(faces) + style + '</style></head><body>' + body + '</body></html>\n')
    record = {"python_markdown": importlib.metadata.version("Markdown"), "fonts": font_records,
              "html_sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "network_resources": 0}
    if args.pdf:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 900}, device_scale_factor=1)
            unexpected = []
            page.on("request", lambda req: unexpected.append(req.url) if req.url.startswith(("http:", "https:")) else None)
            page.goto(target.as_uri(), wait_until="load")
            page.evaluate("document.fonts.ready")
            assert page.locator("img").evaluate_all("xs => xs.every(x => x.complete && x.naturalWidth > 0)")
            assert not unexpected, "Unexpected network resources"
            assert page.locator("figure").count() == 6
            for width in [1280, 390]:
                page.set_viewport_size({"width": width, "height": 900})
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), f"Overflow at {width}px"
            page.emulate_media(media="print")
            for width in [794, 680]:
                page.set_viewport_size({"width": width, "height": 1123})
                assert page.locator("table").evaluate_all("xs => xs.every(x => x.scrollWidth <= x.clientWidth + 2)"), f"Print table overflow at {width}px"
            assert page.locator(".number").evaluate_all("xs => xs.every(x => x.getClientRects().length === 1 && x.scrollWidth <= x.clientWidth + 1)")
            record["qa"] = {"figures_loaded": 6, "screen_widths_without_overflow_px": [1280, 390], "print_tables_without_overflow": True,
                            "short_table_numbers_unbroken": True, "print_table_check_widths_px": [794, 680]}
            page.pdf(path=str(ROOT / "report.pdf"), format="A4", print_background=True, display_header_footer=True,
                     header_template='<span></span>', footer_template='<div style="font:9px Arial;width:100%;text-align:center;color:#64717b">CapsWriter research archive · <span class="pageNumber"></span> / <span class="totalPages"></span></div>')
            record["chromium"] = browser.version
            record["pdf_sha256"] = hashlib.sha256((ROOT / "report.pdf").read_bytes()).hexdigest()
            browser.close()
    (ROOT / "data" / "render.json").write_text(json.dumps(record, indent=2) + "\n")
    files = [p for p in ROOT.rglob("*") if p.is_file() and p.name != "SHA256SUMS" and "__pycache__" not in p.parts]
    files += [main_path, REPO / "docs/en/rx570-research-20261115.md", REPO / "docs/zh-TW/rx570-research-20261115.md",
              REPO / "docs/en/README.md", REPO / "docs/zh-TW/README.md"]
    (ROOT / "SHA256SUMS").write_text("".join(hashlib.sha256(p.read_bytes()).hexdigest() + "  " + str(p.relative_to(REPO)) + "\n" for p in sorted(files)))
    print(json.dumps(record))


if __name__ == "__main__":
    main()
