# FIESTA (Fast IdEntification of State-of-The-Art models using adaptive bandit algorithms)
[![licence](https://img.shields.io/hexpm/l/plug.svg)](https://opensource.org/licenses/Apache-2.0) [![Build Status](https://travis-ci.org/apmoore1/fiesta.svg?branch=master)](https://travis-ci.org/apmoore1/fiesta) [![codecov](https://codecov.io/gh/apmoore1/fiesta/branch/master/graph/badge.svg)](https://codecov.io/gh/apmoore1/fiesta)

## Quick links:
1. [Documentation](https://apmoore1.github.io/fiesta/) - You can find the motivation of the project code base there as well.
2. [Tutorials](https://apmoore1.github.io/fiesta/#tutorials)
3. [Citing](#citing)

## Installing
Requires Python 3.6.1 or greater.

`pip install fiesta-nlp`

## Experiments in the paper
### NER experiments
The code used to create the NER results can be founder [here](https://github.com/apmoore1/NER) with all of the instructions on:
1. How the data was split.
2. How to re-run the models.
3. How the images in the paper were created.
4. Links to all of the original F1 results and data splits.

### Target Dependent Sentiment Analysis experiments
The 500 Macro F1 results from the 12 different TDSA models can be found within [`test_f1.json` file](./results/TDSA/test_f1.json). For replication purposes we have created a [Google Colab notebook](https://github.com/apmoore1/fiesta/blob/master/notebooks/Advantages_of_Model_Selection.ipynb) which can be found here that shows how the results from the paper can be replicated. Further more this notebook is a good example of how to use the `fiesta` library when you already have results and do not need to evaluate any modles.

## Citing
(This will be updated when the ACL version of the paper is published)

If you use FIESTA in your research, please cite [FIESTA: Fast IdEntification of State-of-The-Art models using adaptive bandit algorithms](https://arxiv.org/pdf/1906.12230.pdf)
```
@inproceedings{moss-etal-2019-fiesta,
    title = "{FIESTA}: Fast {I}d{E}ntification of State-of-The-Art models using adaptive bandit algorithms",
    author = "Moss, Henry  and
      Moore, Andrew  and
      Leslie, David  and
      Rayson, Paul",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1281",
    pages = "2920--2930",
}
```

## General Acknowledgments
This code base and it's related FIESTA paper could not have been done without:
1. [Henry Moss's](https://www.lancaster.ac.uk/maths/people/henry-moss) time funded through EPSRC Doctoral Training Grant and the STOR-i Centre for Doctoral Training.
2. [Andrew Moore's](https://apmoore1.github.io/) time funded through EPSRC Doctoral Training Grant.
3. [Paul Rayson's](https://www.lancaster.ac.uk/staff/rayson/) and [David Leslie's](https://www.lancaster.ac.uk/people-profiles/david-leslie) time.
4. Resources -- The loan of a NVIDIA GP100-equipped workstation from [Dr Chris Jewell](https://chicas.lancaster-university.uk/people/jewell.html) at the [Centre for Health Informatics, Computing, and Statistics, Lancaster University](https://chicas.lancaster-university.uk/).
5. We lastly thank the comments and advise of the reviewers from ACL 2019 which has greatly improved the paper.

## Issue template Acknowledgment
We copied/adapted the issues templates from the [allennlp](https://github.com/allenai/allennlp) project.
