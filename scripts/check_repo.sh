#!/usr/bin/env bash
# Refuse a commit that would publish match data, outputs, big files or notebook outputs.
#
#   scripts/check_repo.sh        # checks what is currently staged; exit 1 lists every problem
#
# Blocked: anything under data/, any .csv/.mp4/.xml/.txt (requirements*.txt excepted), any staged
# file over 1 MB, and any .ipynb whose staged copy still has cell outputs.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON="${PYTHON:-python}"
MAX_BYTES=$((1024 * 1024))
fail=0
complain() { echo "check_repo: $1" >&2; fail=1; }

while IFS= read -r -d '' path; do
    lower=$(printf '%s' "$path" | tr '[:upper:]' '[:lower:]')
    base=${lower##*/}

    case "$path" in
        data/*) complain "under data/: $path" ;;
    esac

    case "$base" in
        requirements*.txt) ;;
        *.csv | *.mp4 | *.xml | *.txt) complain "blocked file type: $path" ;;
    esac

    size=$(git cat-file -s ":$path")
    if [ "$size" -gt "$MAX_BYTES" ]; then
        complain "over 1 MB ($size bytes): $path"
    fi

    if [[ "$base" == *.ipynb ]]; then
        if ! git show ":$path" | "$PYTHON" -c '
import json, sys
cells = json.load(sys.stdin).get("cells", [])
sys.exit(1 if any(c.get("outputs") for c in cells) else 0)
'; then
            complain "notebook has outputs (or is not valid JSON): $path"
        fi
    fi
done < <(git diff --cached --name-only -z --diff-filter=ACMR)

if [ "$fail" -ne 0 ]; then
    echo "check_repo: FAILED - unstage the files above" >&2
    exit 1
fi
echo "check_repo: ok"
