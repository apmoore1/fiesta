# FIESTA (Fast IdEntification of State-of-The-Art models using adaptive bandit algorithms)
[![licence](https://img.shields.io/hexpm/l/plug.svg)](https://opensource.org/licenses/Apache-2.0) [![Build Status](https://travis-ci.org/apmoore1/fiesta.svg?branch=master)](https://travis-ci.org/apmoore1/fiesta)

## Installing
Currently only been test with python version 3.6.1 and above.

`pip install fiesta-nlp`

To run the notebooks as well requires installing the following dependencies through the [requirements](./requirements.txt) file:

`pip install -r requirements.txt`

## Experiments in the paper
### Target Dependent Sentiment Analysis experiments
The 500 Macro F1 results from the 12 different TDSA models can be found within [`test_f1.json` file](./results/TDSA/test_f1.json). For replication purposes we have created a Google Colab notebook which can be found here that shows how the results from the paper can be replicated. Further more this notebook is a good example of how to use the `fiesta` library when you already have results and do not need to evaluate any modles.


## Contributing
We welcome any contributions. If you would like to contribute via a feature request, ask a question, or report a bug please submit an issue and we will work with you to incorportate this into the fiesta package. We would appreciate that all pull request originate/start from an issue.

### Requirements file's explained
The [requirements](./requirements.txt) stores all of the dependencies to ensure the whole package will run and tests **as well as the notebooks**.

The [package requirements](./package_requirements.txt) stores only the requirements needed for the package and tests and is therefore used by travis to ensure the tests will pass. This also stores the *pylint* dependency which is the linter used for this package.

### Linter
We use the *pylint* linter of which we would like all contributions to adhere/keep to this linter.

### Issue template Acknowledgment
We copied/adapted the issues templates from the [allennlp](https://github.com/allenai/allennlp) project.