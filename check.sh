#!/usr/bin/env bash
# Single quality gate used by CI and by build-lab before every push.
set -euo pipefail
cd "$(dirname "$0")"
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --release

# Wasm smoke test (needs the wasm32 target and Node); skipped locally if absent, required in CI.
if rustup target list --installed 2>/dev/null | grep -q wasm32-unknown-unknown && command -v node >/dev/null; then
  bash web/test/run-smoke.sh
elif [ -n "${CI:-}" ]; then
  echo "wasm target or node missing in CI" >&2; exit 1
else
  echo "skipping wasm smoke test (no wasm32 target or node)"
fi
