#!/usr/bin/env python3
"""Extract an allowlisted Nov10 research dataset without changing source evidence.

Usage: python extract_historical.py NOV10_ROOT OUTPUT_JSON
Only standard-library analysis runs; no imports from the retained experiment.
"""
import argparse
import collections
import hashlib
import json
import math
import statistics
import unicodedata
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def units(text, language, mode):
    if mode == "raw":
        return list(text) if language == "zh" else text.split()
    text = unicodedata.normalize("NFKC", text).casefold()
    if language == "zh":
        return list("".join(c for c in text if c.isalnum()))
    text = text.replace("'", "").replace("’", "")
    return "".join(c if c.isalnum() else " " for c in text).split()


def distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        nxt = [i]
        for j, y in enumerate(b, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j - 1] + (x != y)))
        row = nxt
    return row[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nov10_root", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()
    root = args.nov10_root
    names = [
        "results/raw.jsonl", "results/telemetry.jsonl", "results/summary.json",
        "results/audit.json", "results/latency-items.csv", "results/paired-quality-items.csv",
        "results/environment.json", "results/model-validation.json",
        "results/deployed-payload-validation.json", "manifest.json", "probes.json",
        "schedule.json", "sources/dataset-downloads.json", "sources/tensor-hashes.json",
        "analyze.py", "finalize_evidence.py", "VERSIONS.txt",
    ]
    sources = [{"root_label": "NOV10", "file": n, "sha256": sha(root / n),
                "bytes": (root / n).stat().st_size} for n in names]
    load = lambda name: json.loads((root / name).read_text())
    summary, audit = load("results/summary.json"), load("results/audit.json")
    raw = [json.loads(line) for line in (root / "results/raw.jsonl").read_text().splitlines()]
    sample_rows = [(i, r) for i, r in enumerate(raw, 1) if r["type"] == "sample"]
    measured = [(i, r) for i, r in sample_rows if not r.get("warmup")]
    assert len(measured) == audit["measured_calls"] == summary["total_calls"] == 860
    assert len(sample_rows) == 874 and len(raw) == 881
    telemetry = [json.loads(line) for line in (root / "results/telemetry.jsonl").read_text().splitlines()]
    def latency_exclusions(r):
        if r["phase"] not in ["latency", "long"]:
            return []
        reasons = []
        if r["status"] != "ok":
            reasons.append("call_error")
        if r.get("aborted"):
            reasons.append("abort")
        if r.get("context_error"):
            reasons.append("context_error")
        samples = [s for s in telemetry if r["start_unix_s"] - 1 <= s["unix_s"] <= r["end_unix_s"] + 1]
        if not samples:
            reasons.append("telemetry_missing")
        for s in samples:
            try:
                if float(s["temperature.gpu"]) >= 87:
                    reasons.append("temperature>=87")
                if float(s["clocks.current.graphics"]) < 700:
                    reasons.append("clock<700")
                if s["clocks_throttle_reasons.sw_thermal_slowdown"] == "Active" or s["clocks_throttle_reasons.hw_thermal_slowdown"] == "Active":
                    reasons.append("thermal_slowdown")
            except (KeyError, ValueError):
                reasons.append("telemetry_missing")
        return sorted(set(reasons))
    observations = []
    scalars = ["job", "variant", "block", "ordinal", "id", "phase", "audio_s", "audio_sha256",
               "language", "bin_s", "repeat", "warmup", "wall_s", "status", "input_samples",
               "consumed_samples", "seed", "temperature", "frontend_s", "backend_s", "encoder_s",
               "prompt_s", "prompt_positions", "decode_s", "empty", "aborted", "retries",
               "generation_limit_hit", "context_error", "base_id"]
    for line, r in sample_rows:
        row = {k: r[k] for k in scalars if k in r}
        row["source"] = {"file": "results/raw.jsonl", "line": line}
        row["text_sha256"] = hashlib.sha256(r["text"].encode()).hexdigest()
        row["latency_exclusions"] = latency_exclusions(r)
        row["prefill_s"] = sum(a["prefill_s"] for a in r.get("attempts", []))
        row["generation_s"] = sum(a["generation_s"] for a in r.get("attempts", []))
        row["output_tokens"] = sum(a["n_generate"] for a in r.get("attempts", []))
        if r.get("reference") and r.get("language"):
            row["edit_metrics"] = {}
            for mode in ["raw", "normalized"]:
                ref, hyp = units(r["reference"], r["language"], mode), units(r["text"], r["language"], mode)
                row["edit_metrics"][mode] = {"reference_units": len(ref), "errors": distance(ref, hyp)}
        observations.append(row)
    pairs = [{k: v for k, v in p.items() if k != "disagreements"} for p in summary["pairs"]]
    verified = []
    for p in pairs:
        if p["phase"] != "quality" or p["source"] is not None or p["mode"] == "script":
            continue
        groups = {v: {r["id"]: r for r in observations if r["variant"] == v and r["phase"] == "quality"
                      and r.get("language") == p["language"] and r["status"] == "ok"
                      and not r.get("context_error") and not r.get("aborted")} for v in [p["a"], p["b"]]}
        ids = sorted(groups[p["a"]].keys() & groups[p["b"]].keys())
        a = [groups[p["a"]][i]["edit_metrics"][p["mode"]] for i in ids]
        b = [groups[p["b"]][i]["edit_metrics"][p["mode"]] for i in ids]
        assert len(ids) == p["n"]
        assert sum(x["errors"] for x in a) == p["a_errors"]
        assert sum(x["errors"] for x in b) == p["b_errors"]
        assert sum(x["reference_units"] for x in a) == p["reference_units"]
        verified.append({"metric": p["language"] + "/" + p["mode"], "n": len(ids)})
    for aggregate in summary["latency"]:
        rows = [r for r in observations if not r.get("warmup") and r["variant"] == aggregate["variant"]
                and r["phase"] in ["latency", "long"] and r["bin_s"] == aggregate["bin_s"]
                and r["status"] == "ok" and (aggregate["kind"] == "all_completed" or not r["latency_exclusions"])]
        assert len(rows) == aggregate["n"]
        walls = sorted(r["wall_s"] for r in rows)
        assert math.isclose(statistics.median(walls), aggregate["p50_s"], abs_tol=1e-14)
        assert walls[math.ceil((len(walls) - 1) * 0.95)] == aggregate["p95_s"]
    # Existing CIs remain attributed to the retained NumPy analysis; this extractor
    # independently recomputes edit counts and wall quantiles, never invents new CIs.
    result = {
        "schema": 1, "study": "Nov10 P4 Q5/Q6 retained-data experiment", "source_root_label": "NOV10",
        "source_files": sources,
        "audit": {k: audit[k] for k in ["metadata_records", "sample_records_including_warmups", "warmups",
                  "measured_calls", "planned_measured_calls", "missing_scheduled_calls", "phase_counts",
                  "quality_denominators", "latency_exclusion_count", "latency_exclusion_reasons",
                  "telemetry_compute_process_count_samples", "telemetry_process_scope", "variants"]},
        "pairs": pairs, "latency": summary["latency"], "paired_latency": summary["paired_latency"],
        "condition_effects": summary["condition_effects"], "sampling_variation": summary["sampling_variation"],
        "anomalies": summary["anomalies"],
        "failures": [{k: r.get(k) for k in ["job", "variant", "id", "phase", "status", "aborted",
                      "retries", "generation_limit_hit", "context_error"]} for r in summary["failures"]],
        "model_validation": load("results/model-validation.json"),
        "deployed_payload_validation": load("results/deployed-payload-validation.json"),
        "observations": observations,
        "fixtures": [{**{k: row[k] for k in ["id", "phase", "language", "audio_s", "audio_sha256",
                      "original_audio_sha256", "source", "repo", "revision", "config", "split", "row_index",
                      "source_id", "source_audio_path", "license", "reference_chars", "reference_words", "tags"] if k in row},
                      "reference_sha256": hashlib.sha256(row["reference"].encode()).hexdigest(),
                      "source_selector": {"file": "manifest.json", "json_pointer": "/" + str(i)}}
                     for i, row in enumerate(load("manifest.json"))],
        "verification": {"measured_count": 860, "raw_lines": 881, "quality_counts_recomputed": verified,
                         "latency_aggregates_recomputed": len(summary["latency"]),
                         "bootstrap_intervals": "Copied exactly from retained summary; not recomputed by this stdlib extractor."},
    }
    # Model-validation artifacts use task-root paths in model entries. Retain only
    # basename labels; all other fields above are strictly allowlisted.
    for model in result["model_validation"].get("models", []):
        if "path" in model:
            model["path"] = Path(model["path"]).name
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    for forbidden in ["/home/", "/mnt/", "/app/", "192.168.", "10.0."]:
        assert forbidden not in rendered, forbidden
    args.output_json.write_text(rendered)
    print(json.dumps({"output": args.output_json.name, "bytes": len(rendered.encode()),
                      "sha256": sha(args.output_json), "measured_calls": 860,
                      "latency_aggregates_verified": len(summary["latency"])}))


if __name__ == "__main__":
    main()
