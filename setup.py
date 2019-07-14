from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(name='fiesta_nlp',
      version='0.0.1',
      description='Fast IdEntification of State-of-The-Art models using adaptive bandit algorithms',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/apmoore1/fiesta',
      author='Andrew Moore, Henry Moss',
      author_email='andrew.p.moore94@gmail.com',
      license='Apache License 2.0',
      install_requires=[
          'numpy',
          'scipy'
      ],
      python_requires='>=3.6.1',
      packages=find_packages(),
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Programming Language :: Python :: 3.6'
      ])