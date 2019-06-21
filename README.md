# FIESTA (Fast IdEntification of State-of-The-Art models using adaptive bandit algorithms)
[![licence](https://img.shields.io/hexpm/l/plug.svg)](https://opensource.org/licenses/Apache-2.0) [![Build Status](https://travis-ci.org/apmoore1/fiesta.svg?branch=master)](https://travis-ci.org/apmoore1/fiesta) [![codecov](https://codecov.io/gh/apmoore1/fiesta/branch/master/graph/badge.svg)](https://codecov.io/gh/apmoore1/fiesta)

## Installing
Currently only been tested with python version 3.6.1 and above.

`pip install fiesta-nlp`

## Experiments in the paper
### NER experiments
The code used to create the NER results can be founder [here](https://github.com/apmoore1/NER) with all of the instructions on:
1. How the data was split.
2. How to re-run the models.
3. How the images in the paper were created.
4. Links to all of the original F1 results and data splits.

### Target Dependent Sentiment Analysis experiments
The 500 Macro F1 results from the 12 different TDSA models can be found within [`test_f1.json` file](./results/TDSA/test_f1.json). For replication purposes we have created a Google Colab notebook which can be found here that shows how the results from the paper can be replicated. Further more this notebook is a good example of how to use the `fiesta` library when you already have results and do not need to evaluate any modles.


## Contributing
We welcome any contributions. If you would like to contribute via a feature request, ask a question, or report a bug please submit an issue and we will work with you to incorportate this into the fiesta package. We would appreciate that all pull request originate/start from an issue.

### Linter
We use the *pylint* linter of which we would like all contributions to adhere/keep to this linter.

### Issue template Acknowledgment
We copied/adapted the issues templates from the [allennlp](https://github.com/allenai/allennlp) project.
