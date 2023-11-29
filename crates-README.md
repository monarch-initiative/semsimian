# semsimian

Semsimian is a package to provide fast semantic similarity calculations for ontologies. 
It is a Rust library with a Python interface. 

This includes implementation of Jaccard and Resnik similarity of terms in an ontology,
as well as a method to calculate the similarity of two sets of terms (so-called
termset similarity). Other methods will be added in the future.

Semsimian is currently integrated into [OAK](https://github.com/INCATools/ontology-access-kit) and
the [Monarch app](https://github.com/monarch-initiative/monarch-app) to provide fast semantic 
similarity calculations.

## Rust Installation

- `cargo add semsimian`

## Python Installation

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

As of version 0.2.11, the semsimian source is released on [GitHub](https://github.com/monarch-initiative/semsimian), with a corresponding set of Python wheels released to [PyPi](https://pypi.org/project/semsimian/) and a corresponding release in [crates.io](https://crates.io/crates/semsimian).

