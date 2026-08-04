#!/usr/bin/env bash
# Lightweight checks that do not require downloading datasets or training.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

echo "[CHECK] Python syntax"
python -m compileall -q code scripts

echo "[CHECK] Shell syntax"
while IFS= read -r script; do
  bash -n "${script}"
done < <(find code scripts -type f -name '*.sh' -print | sort)

echo "[CHECK] Hard-coded local workspace paths"
if rg -n '/workspace/|/root/|/home/' code scripts \
  --glob '*.py' --glob '*.sh' --glob '!scripts/check_repo.sh'; then
  echo "Hard-coded local path found." >&2
  exit 1
fi

echo "[CHECK] Common secret patterns outside local data/results"
if rg -n -i --hidden \
  --glob '!.git/**' \
  --glob '!data/**' \
  --glob '!data_sub/**' \
  --glob '!data_mendeley/**' \
  --glob '!results/**' \
  --glob '!scripts/check_repo.sh' \
  '((api[_-]?key|secret|password|passwd)[[:space:]]*[:=][[:space:]]*[^[:space:]]{8,}|BEGIN [A-Z ]*PRIVATE KEY|ghp_[A-Za-z0-9]{20,}|github_pat_)' .; then
  echo "Potential secret found; review before publishing." >&2
  exit 1
fi

echo "[CHECK] Unignored files at or above GitHub's 100 MiB object limit"
large_unignored=0
while IFS= read -r -d '' file; do
  if [ -f "${file}" ]; then
    size="$(stat -c '%s' "${file}")"
    if [ "${size}" -ge 104857600 ]; then
      printf '%s\t%s bytes\n' "${file}" "${size}" >&2
      large_unignored=1
    fi
  fi
done < <(git ls-files --cached --others --exclude-standard -z)
if [ "${large_unignored}" -ne 0 ]; then
  exit 1
fi

echo "[DONE] Repository checks passed."
