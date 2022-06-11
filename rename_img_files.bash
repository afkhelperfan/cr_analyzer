n=0; ls -tr | while read i; do n=$((n+1)); mv -- "$i" "$(printf '%d' "$n").png"; done
