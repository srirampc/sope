#!/bin/bash

NP=4
export RUST_LOG=info
for x in examples/sope_*.rs; do
    exe_loc=target/debug/examples/$(basename "$x" .rs)
    cmd="mpirun -np $NP $exe_loc"
    echo "Running $cmd"
    $cmd
done
