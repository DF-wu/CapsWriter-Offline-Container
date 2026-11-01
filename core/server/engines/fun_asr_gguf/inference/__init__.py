from .. import logger, console
import os

try:
    from ...llama import llama
except Exception:
    if os.environ.get("CAPSWRITER_LLAMA_BACKEND") == "cuda":
        raise
    ...
