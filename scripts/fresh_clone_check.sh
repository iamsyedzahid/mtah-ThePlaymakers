#!/usr/bin/env bash
# Fresh-clone check (R5): can a stranger install this repository and get the same file?
#
#   bash scripts/fresh_clone_check.sh           # clone the pushed branch, install, test, run twice
#   bash scripts/fresh_clone_check.sh --real    # also run.py --real inside the clone, from raw data
#                                               # (needs $MTAH_DATA_DIR: run it yourself, never in an AI tool)
#
# Clones from the GitHub remote, which is what the organisers will see, not from this folder, into
# outputs/fresh-clone/ (gitignored, deleted first). The new environment uses ${PYTHON:-python}.
# Prints totals and hashes only.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
trap 'echo "fresh_clone_check: FAILED (see above)" >&2' ERR

REAL=0
for arg in "$@"; do
    case "$arg" in
        --real) REAL=1 ;;
        -h | --help) sed -n '2,10p' "$0"; exit 0 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done
if [ "$REAL" = 1 ] && [ -z "${MTAH_DATA_DIR:-}" ]; then
    echo "--real needs MTAH_DATA_DIR set to the folder that holds tactical/" >&2
    exit 2
fi

PYTHON="${PYTHON:-python}"
URL="$(git remote get-url origin)"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
LOCAL_HEAD="$(git rev-parse HEAD)"
DEST="outputs/fresh-clone"

# 1. A clean clone of the pushed branch.
rm -rf "$DEST"
git clone --quiet --branch "$BRANCH" --single-branch "$URL" "$DEST"
CLONE_HEAD="$(git -C "$DEST" rev-parse HEAD)"
echo "cloned $BRANCH @ ${CLONE_HEAD:0:7}"
if [ "$CLONE_HEAD" != "$LOCAL_HEAD" ]; then
    echo "note: your local branch is at ${LOCAL_HEAD:0:7}; push first to check what you have" >&2
fi

# 2. Nothing gated is tracked. The organisers' public CSVs in evaluation/gt and evaluation/results
#    ship with the starter repository on purpose.
bad=$(git -C "$DEST" ls-files \
      | grep -Ev '^data/README\.md$|^requirements[^/]*\.txt$|^evaluation/(gt|results)/' \
      | grep -Ei '^data/|^outputs/|\.(csv|mp4|xml|txt|npz|pkl|pdf)$' || true)
if [ -n "$bad" ]; then
    echo "FAIL: tracked files that must not be public:" >&2
    echo "$bad" >&2
    exit 1
fi
echo "tracked files: no data, outputs, caches or documents"

# 3. Install exactly the pinned requirements into a new environment.
cd "$DEST"
"$PYTHON" -m venv .venv
if [ -x .venv/Scripts/python.exe ]; then VPY=.venv/Scripts/python.exe; else VPY=.venv/bin/python; fi
echo "$("$VPY" --version 2>&1), installing requirements.txt ..."
"$VPY" -m pip install --quiet --disable-pip-version-check -r requirements.txt

# 4. Tests, then the synthetic pipeline twice: the two files must be byte-identical.
"$VPY" -m pytest -q 2>&1 | tail -1
"$VPY" run.py --out outputs/one.csv > /dev/null
"$VPY" run.py --out outputs/two.csv > /dev/null
hash() { "$VPY" -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$1"; }
if cmp -s outputs/one.csv outputs/two.csv; then
    echo "synthetic run twice: identical, sha256 $(hash outputs/one.csv)"
else
    echo "FAIL: two synthetic runs differ" >&2
    exit 1
fi

# 5. Optional: the real submission from raw data, with no cache to lean on.
if [ "$REAL" = 1 ]; then
    echo "real run from raw data (builds a new cache, a few minutes) ..."
    "$VPY" run.py --real --out outputs/real.csv | grep '^sha256'
    rm -rf outputs/cache outputs/real.csv  # leave no copy of the match behind in the clone
fi
echo "fresh_clone_check: ok"
