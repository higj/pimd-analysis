# LAMMPS PIMD Analysis tool
![CodeFactor Grade](https://www.codefactor.io/repository/github/higj/pimd-analysis/badge)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/higj/pimd-analysis)

## What is it?

**LammpsAnalyzer** is a Python class that parses LAMMPS output 
files of both distinguishable and bosonic PIMD simulations. 
It automatically handles physical units, determines the physical parameters 
of the system and provides methods for calculating various energy estimators. 
The class is designed to simplify the analysis of PIMD simulations 
and provide accurate and reliable results.

## Dependencies
- [NumPy - For performing mathematical operations on arrays](https://www.numpy.org)
- [Pint - For manipulating physical quantities](https://pint.readthedocs.io/en/stable/)
- [emcee - Used for autocorrelation analysis](https://emcee.readthedocs.io/en/stable/)

## How to use
Simply import the class and provide the path to the output folder of the simulation, e.g.,

```python
from helper import LammpsAnalyzer

sim = LammpsAnalyzer(path='/path/to/lammps/output/files')
```

The class includes thorough documentation, enabling users to explore 
and utilize its methods to suit their requirements.
