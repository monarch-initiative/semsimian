# semsimian

## Installation

- Set up your virtual environment of choice.
- cd `semsimian` (home directory of this project)
- `pip install maturin`
- `maturin develop`
- `python`
```
Python 3.9.16 (main, Jan 11 2023, 10:02:19) 
[Clang 14.0.6 ] :: Anaconda, Inc. on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> from semsimian import Semsimian
>>> s = Semsimian([('banana', 'is_a', 'fruit'), ('cherry', 'is_a', 'fruit')])
>>> s.jaccard_similarity('banana', 'cherry')
```
This should yield a value of 1.0.


## Releases

As of version 0.1.14, the semsimian source is released on GitHub, with a corresponding set of Python wheels released to Pypi.

To trigger a new set of builds, first update the version number in `Cargo.toml`, then [create a new release](https://github.com/monarch-initiative/semsimian/releases/new).

Wheels are prepared for the following environments and architectures:

| OS      | Architectures                                                                            | Python Versions           |
|---------|------------------------------------------------------------------------------------------|---------------------------|
| Linux   | x86_64, x86_64-unknown-linux-musl, aarch64-unknown-linux-gnu, aarch64-unknown-linux-musl | 3.7, 3.8, 3.9, 3.10, 3.11 |
| MacOS   | x86_64, universal2                                                                       | 3.7, 3.8, 3.9, 3.10, 3.11 |
| Windows | x86_64                                                                                   | 3.7, 3.8, 3.9, 3.10, 3.11 |

## Troubleshooting

### Building for Mac ARM M1 architectures

If a `import semsimian` results in a `ImportError` warning about incompatible architecture, try the following:
- Install `conda`. [This guide may be helpful.](https://towardsdatascience.com/how-to-manage-conda-environments-on-an-apple-silicon-m1-mac-1e29cb3bad12)
- Set up a virtual environment with `conda` so that your Python build is aligned with your processor architecture (in this case, ARM).
Try something like:
```
$ conda create -n myenv python=3.9
...setup happens...
$ conda activate myenv
```
and then proceed as above.


### Code Coverage via Docker

Build a docker image:

```
docker build -t my-rust-app .
```

Run your tests inside a Docker container and generate coverage:
```
docker run -v "$(pwd)":/usr/src/app -t my-rust-app bash -c "CARGO_INCREMENTAL=0 RUSTFLAGS='-Zprofile -Ccodegen-units=1 -Cinline-threshold=0 -Coverflow-checks=off -Zpanic_abort_tests -Cpanic=abort' cargo test && grcov . -s . --binary-path ./target/debug/ -t html --branch --ignore-not-existing -o ./target/debug/coverage/"
```

Get Coverage report from:
```
open ./target/debug/coverage/index.html

```
