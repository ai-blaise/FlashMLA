#!/bin/bash
# Wrapper: patch .version 8.8 -> .version 9.2 in PTX file, then call real ptxas 13.3.
# nvcc calls ptxas with PTX as input. We patch on the fly.
for arg in "$@"; do
    if [[ "$arg" == *.ptx ]] && [ -f "$arg" ]; then
        sed -i 's/\.version 8\.[0-9]*/\.version 9.2/' "$arg"
    fi
done
exec /usr/local/lib/python3.12/dist-packages/nvidia/cu13/bin/ptxas "$@"
