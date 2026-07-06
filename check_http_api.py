#!/usr/bin/env python3
"""CapsWriter HTTP API 诊断工具 — 检查 OpenAI 相容 API 是否正常运作。

用法: python check_http_api.py [--host 127.0.0.1] [--port 6017]
                            [--audio test.wav] [--expect 你好] [--key sk-xxx]
"""

import os, sys, json, shutil, argparse, urllib.request, urllib.error

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 6017


def green(s):
    return f"\033[92m{s}\033[0m"


def red(s):
    return f"\033[91m{s}\033[0m"


def yellow(s):
    return f"\033[93m{s}\033[0m"


def bold(s):
    return f"\033[1m{s}\033[0m"


def compact_for_match(s):
    return "".join(s.split()).casefold()


def check(label):
    sys.stdout.write(f"  {label:40s} ")
    sys.stdout.flush()


def ok(msg="OK"):
    print(green(msg))


def fail(msg):
    print(red(f"FAIL: {msg}"))


def _headers(api_key):
    h = {}
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    return h


def _api_get(base, path, api_key):
    req = urllib.request.Request(f"{base}{path}", headers=_headers(api_key))
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def _api_post(base, path, audio_path, fmt, api_key):
    boundary = "----FormBoundary7MA4YWxkTrZu0gW"
    with open(audio_path, "rb") as f:
        audio_data = f.read()
    body = (
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{os.path.basename(audio_path)}"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n"
        ).encode()
        + audio_data
        + (
            f"\r\n--{boundary}\r\n"
            f'Content-Disposition: form-data; name="model"\r\n\r\nwhisper-1\r\n'
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="response_format"\r\n\r\n{fmt}\r\n'
            f"--{boundary}--\r\n"
        ).encode()
    )
    h = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    h.update(_headers(api_key))
    req = urllib.request.Request(f"{base}{path}", data=body, headers=h)
    with urllib.request.urlopen(req, timeout=120) as r:
        ct = r.headers.get("Content-Type", "")
        raw = r.read()
        if "application/json" in ct:
            return json.loads(raw)
        return {"_raw_text": raw.decode("utf-8"), "_content_type": ct}


def main():
    p = argparse.ArgumentParser(description="CapsWriter HTTP API 诊断工具")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--audio", help="测试用音频文件 (wav/mp3)")
    p.add_argument("--expect", help="转录结果中应包含的文字；未设置时只检查非空")
    p.add_argument(
        "--key",
        default=os.environ.get("CAPSWRITER_HTTP_API_KEY", ""),
        help="API key (或设置环境变量 CAPSWRITER_HTTP_API_KEY)",
    )
    args = p.parse_args()
    api_key = args.key or ""
    base = f"http://{args.host}:{args.port}"
    errors = 0

    print(bold(f"\nCapsWriter HTTP API 诊断 — {base}"))
    if api_key:
        print(f"  API key: {green('已设置')} (长度={len(api_key)})")
    print()

    check("系统 ffmpeg")
    if shutil.which("ffmpeg"):
        ok()
    else:
        fail("ffmpeg 未安装 — 非 raw PCM 音频会上传失败 (500)")
        errors += 1

    check("GET /health")
    try:
        data = _api_get(base, "/health", api_key)
        ok(f"model={data.get('model', '?')} v{data.get('version', '?')}")
    except urllib.error.URLError as e:
        fail(f"无法连接: {e.reason}")
        print(f"\n{yellow('请确认:')}")
        print(
            f"  1. 启动命令: {bold('python start_server_docker.py')}  (NOT start_server.py)"
        )
        print(f"  2. 环境变量: {bold('CAPSWRITER_HTTP_API_ENABLE=true')}")
        print(f"  3. Docker: 取消 docker-compose.yml 中 HTTP_API 相关注释\n")
        return 1
    except urllib.error.HTTPError as e:
        fail(f"HTTP {e.code}")
        if e.code == 401:
            print(
                f"         → 需要 API key: {bold('python check_http_api.py --key YOUR_KEY')}"
            )
        return 1
    except Exception as e:
        fail(str(e))
        return 1

    check("GET /v1/models")
    try:
        data = _api_get(base, "/v1/models", api_key)
        ok(f"{len(data.get('data', []))} model(s)")
    except urllib.error.HTTPError as e:
        fail(f"HTTP {e.code}")
        if e.code == 401:
            print(f"         → API key 不正确或未提供")
        errors += 1
    except Exception as e:
        fail(str(e))
        errors += 1

    if not args.audio:
        print(f"\n{yellow('未提供 --audio，跳过转录测试。')}")
        print(f"用法: python check_http_api.py --audio test.wav\n")
    elif not os.path.exists(args.audio):
        fail(f"文件不存在: {args.audio}")
        errors += 1
    else:
        for fmt in ("json", "text", "verbose_json"):
            check(f"POST /v1/audio/transcriptions fmt={fmt}")
            try:
                result = _api_post(
                    base, "/v1/audio/transcriptions", args.audio, fmt, api_key
                )
                text = result.get("text") or result.get("_raw_text", "")
                if args.expect and compact_for_match(args.expect) not in compact_for_match(
                    text
                ):
                    fail(f"未包含预期文字: {args.expect}")
                    errors += 1
                    continue
                has_chinese = any("\u4e00" <= c <= "\u9fff" for c in text)
                if has_chinese:
                    ok(f"中文 ✓ ({len(text)}字) — {text[:50]}…")
                elif text.strip():
                    ok(f"有文字 ({len(text)}字, 无中文) — {text[:50]}…")
                else:
                    fail("返回空文字 — 音频可能无语音或模型未加载")
                    errors += 1
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                fail(f"HTTP {e.code}: {body[:100]}")
                if e.code == 401:
                    print("         → 需要 API key 或 key 不正确")
                elif e.code == 500 and "ffmpeg" in body.lower():
                    print("         → ffmpeg 未安装。运行: apt install ffmpeg")
                errors += 1
            except Exception as e:
                fail(str(e))
                errors += 1

    print()
    if errors:
        print(red(f"诊断完成: {errors} 项失败"))
        return 1
    print(green("诊断完成: 全部通过"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
