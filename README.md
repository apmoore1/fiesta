# FIESTA (Fast IdEntification of State-of-The-Art models using adaptive bandit algorithms)

## Installing
Currently only been test with python version 3.6.1 and above.

`pip install fiesta-nlp`

To run the notebooks as well requires installing the following dependencies through the [requirements](./requirements.txt) file:

`pip install -r requirements.txt`


## Contributing
We welcome any contributions. If you would like to contribute via a feature request, ask a question, or report a bug please submit an issue and we will work with you to incorportate this into the fiesta package. We would appreciate that all pull request originate/start from an issue.

### Requirements file's explained
The [requirements](./requirements.txt) stores all of the dependencies to ensure the whole package will run and tests **as well as the notebooks**.

The [package requirements](./package_requirements.txt) stores only the requirements needed for the package and tests and is therefore used by travis to ensure the tests will pass. This also stores the *pylint* dependency which is the linter used for this package.

### Linter
We use the *pylint* linter of which we would like all contributions to adhere/keep to this linter.

### Issue template Acknowledgment
We copied/adapted the issues templates from the [allennlp](https://github.com/allenai/allennlp) project.