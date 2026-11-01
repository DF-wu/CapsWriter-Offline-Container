#!/bin/sh
set -eu
# Kept at b7798: newer llama.cpp releases have incompatible ctypes layouts.
curl -fL --retry 3 -o /tmp/llama.tar.gz \
  https://api.github.com/repos/ggml-org/llama.cpp/tarball/c301172f660a1fe0b42023da990bf7385d69adb4
echo '4420071ba535356cb62f8cd65edb924dee321942d45b7e5e402df1f2906e1031  /tmp/llama.tar.gz' | sha256sum -c -
mkdir -p /src/llama
tar -xzf /tmp/llama.tar.gz -C /src/llama --strip-components=1
cmake -S /src/llama -B /src/build -G Ninja \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 \
  -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON \
  -DGGML_NATIVE=OFF -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF
ln -sf libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
# Compiler-only link stubs: never copy them or this environment to runtime.
LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
  cmake --build /src/build --target llama -j "${CUDA_BUILD_JOBS:-8}"
/usr/local/cuda/bin/cuobjdump --list-elf /src/build/bin/libggml-cuda.so > /src/cubins.txt
grep -q sm_61 /src/cubins.txt
python3 /scripts/package_cuda.py /src/build/bin /out
cp /src/cubins.txt /out/cubins.txt
