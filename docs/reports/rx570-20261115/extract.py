#!/usr/bin/env python3
"""Extract allowlisted observations from a PRIVATE packet; never copy raw logs.

Usage: python extract.py PRIVATE_PACKET_DIRECTORY
Only writes this report's data directory. No network or inference.
"""
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

OUT = Path(__file__).resolve().parent / "data"
SOURCES = {}


def load(root, relative, lines=False):
    path = root / relative
    raw = path.read_bytes()
    key = "INSTALLED/" + relative
    SOURCES[key] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                    "retention": "private raw source; public allowlisted extract only"}
    if lines:
        return [(i + 1, json.loads(line)) for i, line in enumerate(raw.decode().splitlines()) if line.strip()]
    return json.loads(raw)


def ref(relative, selector):
    return {"source": "INSTALLED/" + relative, "selector": selector}


def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def extract(packet):
    root = packet / "evergarden-capswriter-runtime-20261115"
    OUT.mkdir(parents=True, exist_ok=True)
    fixtures = load(root, "baseline-complete/audio-manifest.json")
    for i, row in enumerate(fixtures):
        row["evidence"] = ref("baseline-complete/audio-manifest.json", "/" + str(i))
    write("fixtures.json", fixtures)
    requests, tasks, loads, transport, resources, decoder_placement, configuration = [], [], [], [], [], [], []
    for mode, folder in [("cpu", "baseline-complete"), ("hybrid", "baseline-complete"),
                         ("migraphx", "mgx-resume-complete/evidence")]:
        relative = f"{folder}/{mode}-client.jsonl"
        for line, row in load(root, relative, True):
            evidence = ref(relative, f"line:{line}")
            if row.get("type") == "http_result":
                requests.append({**{k: row[k] for k in ["mode", "index", "name", "phase", "seconds", "rtf", "audio_s", "sha256", "reference_units", "edits"]},
                                 "text": row["result"]["text"], "evidence": evidence})
            elif row.get("type") == "ready":
                loads.append({"mode": mode, "stage": "readiness", "seconds": row["seconds"], "evidence": evidence})
            elif row.get("type") == "placement":
                callback_extract = []
                for index, callback in enumerate(row.get("callback_lines", [])):
                    offload = re.search(r"offloaded (\d+)/(\d+) layers to GPU", callback)
                    buffer = re.search(r"(Vulkan0|Vulkan_Host) model buffer size\s*=\s*([\d.]+) MiB", callback)
                    if offload:
                        callback_extract.append({"offloaded_layers": int(offload[1]), "total_reported_layers": int(offload[2]),
                                                 "selector": f"line:{line}/callback_lines/{index}"})
                    if buffer:
                        callback_extract.append({"buffer": buffer[1], "MiB": float(buffer[2]),
                                                 "selector": f"line:{line}/callback_lines/{index}"})
                decoder_placement.append({"mode": mode, "verified": row["verified"],
                                          "model_buffer_MiB": row["model_buffer_MiB"], "callback_extract": callback_extract,
                                          "evidence": evidence})
            elif row.get("type") in ["websocket_result", "websocket_invalid", "server_stopped", "http_invalid"]:
                # Error messages and callback strings may contain private data.
                transport.append({"mode": mode, "type": row["type"],
                                  **{k: row[k] for k in ["seconds", "close_code", "status", "exit_code"] if k in row}, "evidence": evidence})
        for path in sorted((root / folder).glob(f"{mode}-instrument-*.jsonl")):
            relative = str(path.relative_to(root))
            for line, row in load(root, relative, True):
                evidence = ref(relative, f"line:{line}")
                if row.get("type") in ["encoder_load_and_builtin_warmup", "decoder_model_load", "engine_load"]:
                    loads.append({"mode": mode, "stage": row["type"], "seconds": row["seconds"], "evidence": evidence})
                    if row["type"] == "engine_load" and "config" in row:
                        fields = ["encoder_frontend_fn", "encoder_backend_fn", "llm_fn", "onnx_provider", "llm_use_gpu",
                                  "dml_pad_to", "n_ctx", "chunk_size", "memory_num", "enable_aligner", "n_batch", "n_ubatch",
                                  "flash_attn", "offload_kqv", "n_threads", "n_threads_batch"]
                        strict = {}
                        for graph, session in row.get("strict_sessions", {}).items():
                            options = session["provider_options"]["MIGraphXExecutionProvider"]
                            strict[graph] = {"python_fallback_enabled": session["python_fallback_enabled"],
                                             "disable_cpu_ep_fallback": session["disable_cpu_ep_fallback"],
                                             "provider_options": {k: options[k] for k in ["device_id", "migraphx_fp16_enable", "migraphx_exhaustive_tune"]}}
                        configuration.append({"mode": mode, "config": {k: row["config"][k] for k in fields},
                                              "strict_sessions": strict, "evidence": evidence})
                elif row.get("type") == "task":
                    tasks.append({"mode": mode, "sequence": sum(t["mode"] == mode for t in tasks),
                                  **{k: row[k] for k in ["queue_s", "pipeline_s", "source", "language", "audio_bytes", "mel_s", "mel_shape", "frontend_s", "backend_s", "encoder_s", "embedding_shape", "prompt_s", "prompt_shape", "decoder_s"]},
                                  "text": row["result"]["text"],
                                  "attempts": [{k: a[k] for k in ["stable_tokens", "t_prefill", "t_generate", "n_prefill", "n_generate", "is_aborted"]} for a in row["attempts"]],
                                  "evidence": evidence})
        sample_file = f"{folder}/{'dynamic' if mode == 'migraphx' else mode}-samples.jsonl"
        samples = load(root, sample_file, True)
        metrics = [("temperature_C", "temp_mC", max, 1000), ("total_vram_GiB", "vram_used_bytes", max, 2**30),
                   ("available_RAM_GiB", "mem_available_bytes", min, 2**30), ("summed_task_RSS_GiB", "task_group_rss_sum_bytes", max, 2**30),
                   ("gpu_busy_percent", "gpu_busy_percent", max, 1)]
        for label, key, fn, divisor in metrics:
            line, sample = fn(samples, key=lambda item: item[1][key])
            resources.append({"mode": mode, "metric": label, "aggregation": fn.__name__,
                              "value": sample[key] / divisor, "samples": len(samples),
                              "evidence": ref(sample_file, f"line:{line}/{key}")})
    summary_file = "mgx-resume-complete/evidence/summary.json"
    summary = load(root, summary_file)
    diagnostic = summary["reused_diagnostic"]
    write("diagnostic.json", {"embeddings": diagnostic["embeddings"],
                               "strict_both_graph_execution_verified": diagnostic["strict_both_graph_execution_verified"],
                               "numerical_gate_pass": diagnostic["numerical_gate_pass"],
                               "evidence": ref(summary_file, "/reused_diagnostic")})
    numerical = [{**{k: r[k] for k in ["index", "name", "phase", "embedding_relative_l2", "embedding_cosine", "conservative_numerical_gate_pass", "exact_text_parity_to_hybrid", "exact_decoder_ids_parity_to_hybrid"]},
                  "evidence": ref(summary_file, f"/comparisons/{i}")} for i, r in enumerate(summary["comparisons"])]
    placement = []
    for graph in ["frontend", "backend"]:
        profile = next((root / "mgx-resume-complete/evidence").glob(f"app-*-{graph}_*.json"))
        relative = str(profile.relative_to(root))
        data = load(root, relative)
        events = [(i, e) for i, e in enumerate(data) if e.get("cat") == "Node" and e.get("args", {}).get("provider")]
        counts = {}
        for _, e in events:
            provider = e["args"]["provider"]
            counts[provider] = counts.get(provider, 0) + 1
        placement.append({"graph": graph, "execution_event_counts": counts, "event_indices": [i for i, _ in events],
                          "evidence": ref(relative, "Node events with args.provider; counts are executions, not unique nodes")})
    baseline_file = "baseline-complete/summary.json"
    baseline = load(root, baseline_file)
    parity = [{"index": r["index"], "name": r["name"], "exact_text_parity": r["exact_text_parity"],
               "exact_decoder_id_parity": r["exact_decoder_id_parity"],
               "cpu_encoder_embedding_relative_l2": r["cpu_encoder_embedding_relative_l2"],
               "cpu_encoder_embedding_cosine": r["cpu_encoder_embedding_cosine"],
               "evidence": ref(baseline_file, f"/comparison/{i}")} for i, r in enumerate(baseline["comparison"])]
    provenance_file = "baseline-complete/provenance.json"
    provenance = load(root, provenance_file)
    library_file = "mgx-resume-complete/evidence/decoder-library-sha256.json"
    libraries = load(root, library_file)
    preflight_file = "baseline-complete/preflight.json"
    preflight = load(root, preflight_file)
    report_bytes = (root / "REPORT.md").read_bytes()
    SOURCES["INSTALLED/REPORT.md"] = {"sha256": hashlib.sha256(report_bytes).hexdigest(), "bytes": len(report_bytes),
                                     "retention": "private retained synthesis; selected artifact/configuration claims only"}
    report_lines = report_bytes.decode().splitlines()
    artifact_claims = []
    for i, line in enumerate(report_lines, 1):
        for digest in re.findall(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])", line):
            artifact_claims.append({"sha256": digest, "evidence": ref("REPORT.md", f"line:{i}")})
    write("artifacts.json", {"baseline_provenance": provenance, "baseline_evidence": ref(provenance_file, "/"),
                             "decoder_library_sha256": libraries, "library_evidence": ref(library_file, "/"),
                             "installed_kernel": preflight["kernel"], "kernel_evidence": ref(preflight_file, "/kernel"),
                             "decoder_placement": decoder_placement, "configuration": configuration,
                             "retained_report_artifact_claims": artifact_claims,
                             "host_software_evidence": ref("REPORT.md", "line:5 (retained host/software synthesis)")})
    for name, value in [("requests.json", requests), ("tasks.json", tasks), ("loads.json", loads),
                        ("transport.json", transport), ("resources.json", resources), ("numerical.json", numerical),
                        ("placement.json", placement), ("baseline-parity.json", parity)]:
        write(name, value)
    write("sources.json", SOURCES)
    with (OUT / "http.csv").open("w", newline="") as stream:
        fields = ["mode", "index", "name", "phase", "seconds", "rtf", "audio_s", "edits", "reference_units"]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in requests)
    print(f"Extracted {len(requests)} HTTP observations, {len(tasks)} tasks, {len(numerical)} numerical comparisons; {len(SOURCES)} fingerprinted private sources.")


if __name__ == "__main__":
    extract(Path(sys.argv[1]).resolve())
