#!/usr/bin/env python3
"""Validate public observations and optionally fingerprint PRIVATE source roots.

Offline, standard library only. Never imports a model/runtime or changes inputs.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODES = ["cpu", "hybrid", "migraphx"]


def load(name):
    return json.loads((DATA / name).read_text())


def verify_public():
    requests, tasks, numerical = load("requests.json"), load("tasks.json"), load("numerical.json")
    fixtures = {r["name"]: r for r in load("fixtures.json")}
    assert len(requests) == 66 and len(tasks) == 69 and len(numerical) == 22
    summary = []
    for name, fixture in fixtures.items():
        assert fixture["samples"] / fixture["sample_rate"] == fixture["seconds"]
        values = {}
        for mode in MODES:
            rows = [r for r in requests if r["mode"] == mode and r["name"] == name and r["phase"] == "repeat"]
            assert len(rows) == (3 if name in ["short", "reference"] else 2)
            values[mode] = median(r["seconds"] for r in rows)
        assert values["hybrid"] < values["cpu"] and values["hybrid"] < values["migraphx"]
        summary.append({"fixture": name, "n_repeat": len(rows), "median_seconds": values})
    for mode in MODES:
        rows = sorted([r for r in requests if r["mode"] == mode], key=lambda r: r["index"])
        worker = sorted([t for t in tasks if t["mode"] == mode], key=lambda t: t["sequence"])
        assert len(rows) == 22 and [r["index"] for r in rows] == list(range(22)) and len(worker) == 23
        assert len([t for t in load("transport.json") if t["mode"] == mode]) == 4
        assert next(t for t in load("transport.json") if t["mode"] == mode and t["type"] == "websocket_invalid")["close_code"] == 1008
        assert next(t for t in load("transport.json") if t["mode"] == mode and t["type"] == "http_invalid")["status"] == 400
        for r, t in zip(rows, worker):
            assert r["text"] == t["text"]
            assert r["sha256"] == fixtures[r["name"]]["sha256"]
            assert math.isclose(r["rtf"], r["seconds"] / r["audio_s"], rel_tol=1e-14)
            mel_length = fixtures[r["name"]]["samples"] // 160
            audio_positions = 13 * (mel_length // 100) + math.ceil((mel_length % 100) / 8)
            assert t["mel_shape"] == [128, mel_length]
            assert t["embedding_shape"] == [audio_positions, 2048]
            assert t["prompt_shape"] == [audio_positions + 22, 2048]
            assert all(not a["is_aborted"] and a["n_generate"] < 512 for a in t["attempts"])
    for n in numerical:
        assert n["conservative_numerical_gate_pass"] == (n["embedding_relative_l2"] < .02 and n["embedding_cosine"] > .999)
        assert not n["conservative_numerical_gate_pass"] and n["exact_text_parity_to_hybrid"] and n["exact_decoder_ids_parity_to_hybrid"]
        group = {m: next(t for t in tasks if t["mode"] == m and t["sequence"] == n["index"]) for m in MODES}
        assert group["hybrid"]["text"] == group["migraphx"]["text"]
        assert [a["stable_tokens"] for a in group["hybrid"]["attempts"]] == [a["stable_tokens"] for a in group["migraphx"]["attempts"]]
    assert min(n["embedding_relative_l2"] for n in numerical) == 0.036888849942249974
    assert max(n["embedding_relative_l2"] for n in numerical) == 0.05795516151782084
    parity = load("baseline-parity.json")
    assert sum(r["exact_text_parity"] for r in parity) == 19
    assert all(r["cpu_encoder_embedding_relative_l2"] == 0 and math.isclose(r["cpu_encoder_embedding_cosine"], 1, abs_tol=1e-14) for r in parity)
    assert [r["execution_event_counts"] for r in load("placement.json")] == [
        {"MIGraphXExecutionProvider": 257}, {"MIGraphXExecutionProvider": 24}]
    csv_rows = list(csv.DictReader((DATA / "http.csv").open()))
    assert len(csv_rows) == len(requests)
    assert all(float(c["seconds"]) == r["seconds"] for c, r in zip(csv_rows, requests))
    # Public historical observations retain edit counts and exclusion flags,
    # so wall quantiles and paired primary totals can be rechecked without text.
    history = load("historical.json")
    observations = history["observations"]
    assert len(history["fixtures"]) == 240
    assert len(observations) == 874 and sum(not r.get("warmup") for r in observations) == 860
    assert history["audit"]["missing_scheduled_calls"] == 600
    for aggregate in history["latency"]:
        rows = [r for r in observations if not r.get("warmup") and r["variant"] == aggregate["variant"]
                and r["phase"] in ["latency", "long"] and r["bin_s"] == aggregate["bin_s"]
                and r["status"] == "ok" and (aggregate["kind"] == "all_completed" or not r["latency_exclusions"])]
        assert len(rows) == aggregate["n"]
        walls = sorted(r["wall_s"] for r in rows)
        assert median(walls) == aggregate["p50_s"]
        assert walls[math.ceil((len(walls) - 1) * .95)] == aggregate["p95_s"]
    for p in history["pairs"]:
        if p["phase"] != "quality" or p["source"] is not None or p["mode"] == "script":
            continue
        groups = {v: {r["id"]: r for r in observations if r["variant"] == v and r["phase"] == "quality"
                       and r["language"] == p["language"] and r["status"] == "ok" and not r.get("context_error") and not r.get("aborted")}
                  for v in [p["a"], p["b"]]}
        ids = groups[p["a"]].keys() & groups[p["b"]].keys()
        assert len(ids) == p["n"]
        for variant, field in [(p["a"], "a_errors"), (p["b"], "b_errors")]:
            assert sum(groups[variant][i]["edit_metrics"][p["mode"]]["errors"] for i in ids) == p[field]
        assert sum(groups[p["a"]][i]["edit_metrics"][p["mode"]]["reference_units"] for i in ids) == p["reference_units"]
    live = load("live.json")
    assert [live["e2e"]["profile_execution"][g]["provider_counts"] for g in ["frontend", "backend"]] == [
        {"MIGraphXExecutionProvider": 70}, {"MIGraphXExecutionProvider": 9}]
    assert all(not r["values"]["initial_conservative_gate_pass"] for r in live["e2e"]["numeric"])
    assert len(live["e2e"]["compile_log_intervals"]["events"]) == 8
    main = ROOT.parent / "rx570-qwen3-asr-research-20261115.md"
    prose = main.read_text()
    # Detect stale manually transcribed medians and startup rows.
    for row in summary:
        for value in row["median_seconds"].values():
            assert f"{value:.6f}" in prose, (row["fixture"], value)
    for row in load("loads.json"):
        assert f'{row["seconds"]:.6f}' in prose, row
    for name in ["pipeline", "http-repeat", "http-sequence", "startup-components", "numerical-gate", "q5-q6-quality"]:
        assert f"figures/{name}.png" in prose
        svg = (ROOT / "figures" / (name + ".svg")).read_text()
        assert "<title>" in svg and "<desc>" in svg
        assert (ROOT / "figures" / (name + ".png")).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    privacy_scan()
    return {"installed_http": 66, "installed_tasks": 69, "numerical_pass": "0/22",
            "hybrid_migraphx_text_and_token_parity": "22/22", "cpu_hybrid_text_parity": "19/22",
            "historical_samples": 874, "historical_measured": 860, "historical_latency_aggregates": 23,
            "retained_bootstrap_intervals": "not regenerated; original analysis attributed",
            "repeat_medians": summary}


def privacy_scan():
    candidates = list(ROOT.rglob("*.json")) + list(ROOT.rglob("*.csv")) + list(ROOT.rglob("*.md")) + [ROOT.parent / "rx570-qwen3-asr-research-20261115.md"]
    patterns = [r"/(?:home|Users|data/workspace|mnt)/[^\s`\"']+", r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
                r"(?:github_pat_|ghp_|sk-)[A-Za-z0-9_\-]{16,}", r"(?i)authorization:\s*(?:bearer|basic)\s+\S+"]
    for path in candidates:
        text = path.read_text()
        for pattern in patterns:
            matches = re.findall(pattern, text)
            # Exact four-component package versions retained in provenance.
            matches = [m for m in matches if m not in {"0.0.7798", "7.16.26385"}]
            assert not matches, (path.relative_to(ROOT.parent), pattern, matches[:3])


def verify_sources(args):
    checked = 0
    def check(root, relative, expected):
        nonlocal checked
        path = root / relative
        assert path.is_file(), f"Missing private source: {relative}"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative
        checked += 1
    if args.packet:
        for key, metadata in load("sources.json").items():
            check(args.packet / "evergarden-capswriter-runtime-20261115", key.removeprefix("INSTALLED/"), metadata["sha256"])
    if args.nov10:
        for metadata in load("historical.json")["source_files"]:
            check(args.nov10, metadata["file"], metadata["sha256"])
    for metadata in load("live.json")["sources"].values():
        root = args.live_smoke if metadata["root"] == "LIVE-SMOKE" else args.live_e2e
        if root:
            check(root, metadata["relative_file"], metadata["sha256"])
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["packet", "nov10", "live-smoke", "live-e2e"]:
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    report = verify_public()
    report["private_source_hashes_checked"] = verify_sources(args)
    report["limitations"] = "Public metrics are not raw-array recomputation; full retained profiles/arrays remain private. No new inference."
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
