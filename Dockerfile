FROM rust:latest

WORKDIR /usr/src/app
COPY . .

RUN rustup install nightly
RUN rustup default nightly

RUN apt-get update && apt-get install -y llvm-dev libclang-dev clang

RUN cargo install grcov
