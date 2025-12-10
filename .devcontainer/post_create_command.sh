#!/bin/bash

set -euo pipefail

# Package manager dependencies.
sudo apt update
sudo apt install -y build-essential pkg-config libudev-dev llvm libclang-dev protobuf-compiler libssl-dev vim

# Install the Rust toolchain.
rustup toolchain install
rustup component add rustfmt
rustup component add clippy
