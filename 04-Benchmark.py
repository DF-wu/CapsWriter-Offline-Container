#!/usr/bin/env python3
"""ASR benchmark script for CPU vs GPU comparison."""

from __future__ import annotations

import argparse
import statistics
import time
import wave
from typing import Any

import numpy as np

from config_server import FunASRNanoGGUFArgs, Qwen3ASRGGUFArgs
from util.fun_asr_gguf import create_asr_engine as create_fun_engine
from util.qwen_asr_gguf import create_asr_engine as create_qwen_engine
from util.qwen_asr_gguf.inference.utils import load_audio as load_qwen_audio


def make_audio(seconds: float, sample_rate: int = 16000) -> np.ndarray:
    total = int(seconds * sample_rate)
    t = np.arange(total, dtype=np.float32) / sample_rate

    carrier = (
        0.45 * np.sin(2 * np.pi * 180.0 * t)
        + 0.25 * np.sin(2 * np.pi * 360.0 * t)
        + 0.15 * np.sin(2 * np.pi * 720.0 * t)
    )

    syllable_rate = 4.5
    envelope = 0.5 * (1.0 + np.sin(2 * np.pi * syllable_rate * t))
    envelope = np.clip(envelope, 0.0, 1.0) ** 2

    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, 0.015, total).astype(np.float32)

    audio = carrier * envelope + noise
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
    return audio


def prepare_audio(
    seconds: float, audio_file: str | None, sample_rate: int = 16000
) -> np.ndarray:
    if audio_file:
        if audio_file.lower().endswith(".wav"):
            with wave.open(audio_file, "rb") as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                channels = wav_file.getnchannels()
                src_rate = wav_file.getframerate()
                sample_width = wav_file.getsampwidth()

            if sample_width != 2:
                raise ValueError("Benchmark WAV input must be 16-bit PCM")
            if src_rate != sample_rate:
                raise ValueError(f"Expected {sample_rate}Hz WAV, got {src_rate}Hz")

            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            if channels > 1:
                audio = audio.reshape(-1, channels).mean(axis=1)
            return audio.astype(np.float32)
        return load_qwen_audio(audio_file, sample_rate=sample_rate)
    return make_audio(seconds, sample_rate=sample_rate)


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "min_ms": min(values) * 1000.0,
        "median_ms": statistics.median(values) * 1000.0,
        "mean_ms": statistics.mean(values) * 1000.0,
        "max_ms": max(values) * 1000.0,
    }


def print_summary(label: str, values: list[float]) -> None:
    stats = summarize(values)
    print(
        f"{label}: min={stats['min_ms']:.2f}ms "
        f"median={stats['median_ms']:.2f}ms mean={stats['mean_ms']:.2f}ms "
        f"max={stats['max_ms']:.2f}ms"
    )


def build_qwen_engine(hardware: str, onnx: str | None = None, llama: str | None = None):
    kwargs = {
        k: v for k, v in Qwen3ASRGGUFArgs.__dict__.items() if not k.startswith("_")
    }
    if onnx is not None:
        kwargs["use_cuda"] = onnx == "gpu"
        kwargs["use_dml"] = False
    if llama is not None:
        kwargs["vulkan_enable"] = llama == "gpu"
    kwargs["verbose"] = False
    return create_qwen_engine(**kwargs)


def build_fun_engine(hardware: str):
    kwargs = {
        k: v for k, v in FunASRNanoGGUFArgs.__dict__.items() if not k.startswith("_")
    }
    kwargs["use_cuda"] = hardware == "gpu"
    kwargs["dml_enable"] = False
    kwargs["vulkan_enable"] = hardware == "gpu"
    kwargs["verbose"] = False
    return create_fun_engine(**kwargs)


def qwen_providers(engine: Any) -> dict[str, list[str]]:
    encoder = engine.engine.encoder
    return {
        "frontend": list(encoder.sess_fe.get_providers()),
        "backend": list(encoder.sess_be.get_providers()),
    }


def fun_providers(engine: Any) -> dict[str, list[str]]:
    return {
        "encoder": list(engine.models.encoder.sess.get_providers()),
        "ctc": list(engine.models.ctc_decoder.sess.get_providers()),
    }


def benchmark_qwen(
    seconds: float,
    hardware: str,
    runs: int,
    language: str | None,
    context: str | None,
    onnx: str | None = None,
    llama: str | None = None,
    temperature: float = 0.4,
    rollback_num: int = 5,
    audio_file: str | None = None,
) -> None:
    audio = prepare_audio(seconds, audio_file)

    t0 = time.perf_counter()
    engine = build_qwen_engine(hardware, onnx=onnx, llama=llama)
    init_time = time.perf_counter() - t0

    print(
        f"model=qwen_asr hardware={hardware} preset={Qwen3ASRGGUFArgs.preset} onnx={onnx or ('gpu' if Qwen3ASRGGUFArgs.use_cuda else 'cpu')} llama={llama or ('gpu' if Qwen3ASRGGUFArgs.vulkan_enable else 'cpu')}"
    )
    print(f"providers={qwen_providers(engine)}")
    print(f"audio_samples={len(audio)}")
    print_summary("cold_init", [init_time])

    totals: list[float] = []
    encode: list[float] = []
    prefill: list[float] = []
    decode: list[float] = []
    align: list[float] = []
    prompt_build: list[float] = []
    prompt_static: list[float] = []
    prompt_history_tokenize: list[float] = []
    prompt_history_embed: list[float] = []
    prompt_concat: list[float] = []

    for _ in range(runs):
        t_run = time.perf_counter()
        result = engine.engine.asr(
            audio,
            context=context,
            language=language,
            temperature=temperature,
            rollback_num=rollback_num,
        )
        totals.append(time.perf_counter() - t_run)

        perf = result.performance or {}
        encode.append(float(perf.get("encode_time", 0.0)))
        prefill.append(float(perf.get("prefill_time", 0.0)))
        decode.append(float(perf.get("decode_time", 0.0)))
        align.append(float(perf.get("align_time", 0.0)))
        prompt_build.append(float(perf.get("prompt_build_time", 0.0)))
        prompt_static.append(float(perf.get("prompt_static_time", 0.0)))
        prompt_history_tokenize.append(
            float(perf.get("prompt_history_tokenize_time", 0.0))
        )
        prompt_history_embed.append(float(perf.get("prompt_history_embed_time", 0.0)))
        prompt_concat.append(float(perf.get("prompt_concat_time", 0.0)))

    print_summary("total", totals)
    print_summary("encode", encode)
    print_summary("prompt_build", prompt_build)
    print_summary("prompt_static", prompt_static)
    print_summary("prompt_history_tokenize", prompt_history_tokenize)
    print_summary("prompt_history_embed", prompt_history_embed)
    print_summary("prompt_concat", prompt_concat)
    print_summary("prefill", prefill)
    print_summary("decode", decode)
    print_summary("align", align)
    engine.cleanup()


def benchmark_fun(
    seconds: float,
    hardware: str,
    runs: int,
    language: str | None,
    context: str | None,
    audio_file: str | None = None,
) -> None:
    audio = prepare_audio(seconds, audio_file)

    t0 = time.perf_counter()
    engine = build_fun_engine(hardware)
    init_time = time.perf_counter() - t0

    print(f"model=fun_asr_nano hardware={hardware}")
    print(f"providers={fun_providers(engine)}")
    print(f"audio_samples={len(audio)}")
    print_summary("cold_init", [init_time])

    totals: list[float] = []
    encode: list[float] = []
    ctc: list[float] = []
    prepare: list[float] = []
    inject: list[float] = []
    llm_generate: list[float] = []
    align: list[float] = []

    for _ in range(runs):
        stream = engine.create_stream()
        stream.accept_waveform(engine.sample_rate, audio)

        t_run = time.perf_counter()
        result = engine.decode_stream(
            stream,
            language=language,
            context=context,
            verbose=False,
            reporter=None,
        )
        totals.append(time.perf_counter() - t_run)

        timings = result.timings
        encode.append(float(getattr(timings, "encode", 0.0)))
        ctc.append(float(getattr(timings, "ctc", 0.0)))
        prepare.append(float(getattr(timings, "prepare", 0.0)))
        inject.append(float(getattr(timings, "inject", 0.0)))
        llm_generate.append(float(getattr(timings, "llm_generate", 0.0)))
        align.append(float(getattr(timings, "align", 0.0)))

    print_summary("total", totals)
    print_summary("encode", encode)
    print_summary("ctc", ctc)
    print_summary("prepare", prepare)
    print_summary("inject", inject)
    print_summary("llm_generate", llm_generate)
    print_summary("align", align)
    engine.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare ASR latency on CPU vs GPU.")
    parser.add_argument("--model", choices=["qwen", "fun"], required=True)
    parser.add_argument("--hardware", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--language", default=None)
    parser.add_argument("--context", default=None)
    parser.add_argument("--audio-file", default=None)
    parser.add_argument("--onnx", choices=["cpu", "gpu"], default=None)
    parser.add_argument("--llama", choices=["cpu", "gpu"], default=None)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--rollback-num", type=int, default=5)
    args = parser.parse_args()

    print(f"audio_seconds={args.seconds}")
    print(f"runs={args.runs}")

    if args.model == "qwen":
        benchmark_qwen(
            args.seconds,
            args.hardware,
            args.runs,
            args.language,
            args.context,
            onnx=args.onnx,
            llama=args.llama,
            temperature=args.temperature,
            rollback_num=args.rollback_num,
            audio_file=args.audio_file,
        )
    else:
        benchmark_fun(
            args.seconds,
            args.hardware,
            args.runs,
            args.language,
            args.context,
            audio_file=args.audio_file,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
