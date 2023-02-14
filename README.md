# RustSim

- Setup your virtual environment of choice.
- cd `rustsim` (home directory of this project)
- `pip install maturin`
- `maturin develop`
- `python`
```
Python 3.9.16 (main, Jan 11 2023, 10:02:19) 
[Clang 14.0.6 ] :: Anaconda, Inc. on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import rustsim
>>> rustsim.run("tests/data/test_set.tsv", "tests/data/closures.tsv")
```
should yield
```
TermSetPairwiseSimilarity {
    set_id: "set3",
    original_subject_termset: {
        "apple",
        "orange",
        "banana",
    },
    subject_termset: {
        "fruit",
        "orange",
        "apple",
        "tropical",
        "banana",
    },
    original_object_termset: {
        "mango",
        "banana",
        "cheese",
        "papaya",
        "beef",
    },
    object_termset: {
        "mango",
        "fruit",
        "papaya",
        "banana",
        "cheese",
        "beef",
        "meat",
        "dairy",
        "tropical",
    },
    jaccard_similarity: 0.2727272727272727,
}
TermSetPairwiseSimilarity {
    set_id: "set1",
    original_subject_termset: {
        "apple",
        "orange",
        "banana",
    },
    subject_termset: {
        "fruit",
        "orange",
        "apple",
        "tropical",
        "banana",
    },
    original_object_termset: {
        "apple",
        "orange",
        "banana",
    },
    object_termset: {
        "orange",
        "banana",
        "apple",
        "tropical",
        "fruit",
    },
    jaccard_similarity: 1.0,
}
TermSetPairwiseSimilarity {
    set_id: "set2",
    original_subject_termset: {
        "apple",
        "orange",
        "banana",
    },
    subject_termset: {
        "fruit",
        "orange",
        "apple",
        "tropical",
        "banana",
    },
    original_object_termset: {
        "mango",
        "apple",
        "papaya",
    },
    object_termset: {
        "apple",
        "fruit",
        "papaya",
        "mango",
        "tropical",
    },
    jaccard_similarity: 0.42857142857142855,
}
```

## Troubleshooting

### Building for Mac ARM M1 architectures

If a `import rustsim` results in a `ImportError` warning about incompatible architecture, try the following:
- Install `conda`. [This guide may be helpful.](https://towardsdatascience.com/how-to-manage-conda-environments-on-an-apple-silicon-m1-mac-1e29cb3bad12)
- Set up a virtual environment with `conda` so that your Python build is aligned with your processor architecture (in this case, ARM).
Try something like:
```
$ conda create -n myenv python=3.9
...setup happens...
$ conda activate myenv
```
and then proceed as above.
