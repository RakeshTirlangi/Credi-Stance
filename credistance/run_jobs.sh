#!/usr/bin/env bash
# Run jobs.txt with N parallel workers on one GPU; skip jobs whose output already exists.
cd "$(dirname "$0")/.."
N=${1:-3}
JOBS=${2:-jobs.txt}
mkdir -p logs
grep -v "^$" "$JOBS" | while read -r cmd; do
  out=$(echo "$cmd" | sed -n 's/.*--out \([^ ]*\).*/\1/p')
  if [ -f "$out/config.json" ]; then continue; fi
  echo "$cmd"
done | xargs -d '\n' -P "$N" -I{} bash -c 'out=$(echo "{}" | sed -n "s/.*--out \([^ ]*\).*/\1/p"); mkdir -p "$out"; {} > "logs/$(echo $out | tr / _).log" 2>&1; echo "finished $out"'
echo ALL_DONE
