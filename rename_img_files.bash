#!/bin/bash

args=("$@")
for j in `seq ${args[1]} ${args[2]}` ; do
    echo ${j}
    n=0; ls -tr data/${args[0]}/$j | while read i; do n=$((n+1)); mv -- "data/${args[0]}/$j/$i" "data/${args[0]}/$j/$(printf '%d' "$n").png"; mv "data/${args[0]}/$j/7.png" "data/${args[0]}/$j/tree.json"; done
done

