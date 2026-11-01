"""Package the locally built libraries with integrity data for the downloader."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile


def package(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    libraries = sorted(p for p in source.glob("*.so*") if p.is_file())
    if not (source / "libggml-cuda.so").is_file():
        raise RuntimeError("CUDA backend library missing")
    manifest = {}
    archive = output / "llama-b7798-cuda-sm61.tar.gz"
    with tarfile.open(archive, "w:gz", dereference=True) as tar:
        for library in libraries:
            data = library.read_bytes()
            manifest[library.name] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            tar.add(library, arcname=library.name)
    metadata = {
        "source_commit": "c301172f660a1fe0b42023da990bf7385d69adb4",
        "source_archive_sha256": "4420071ba535356cb62f8cd65edb924dee321942d45b7e5e402df1f2906e1031",
        "cuda_architectures": "61", "cuda_toolkit": "11.8",
        "archive_size": archive.stat().st_size,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "libraries": manifest,
    }
    (output / "runtime.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    package(Path(sys.argv[1]), Path(sys.argv[2]))
