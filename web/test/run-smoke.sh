#!/usr/bin/env bash
# Build the wasm module and run the Node smoke test against native output.
set -euo pipefail
cd "$(dirname "$0")/../.."
cargo build --release -p foretone-wasm --target wasm32-unknown-unknown
cargo run --release -q -p foretone-wasm --example reference > target/native-reference.txt
node web/test/smoke.mjs target/wasm32-unknown-unknown/release/foretone_wasm.wasm target/native-reference.txt
