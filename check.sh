#!/usr/bin/env bash
# Single quality gate used by CI and by build-lab before every push.
set -euo pipefail
cd "$(dirname "$0")"
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --release
