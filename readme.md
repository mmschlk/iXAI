<p align="center">
  <!-- License -->
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-brightgreen.svg" alt="mit_license">
  </a>
</p>

# Increment Explain 

This is the first iteration of our incremental explainer package.

Currently, it includes two explanation methods: PFI and SAGE.

Please look at `pandas_example.py`, `neural_network.py`, `agrawal_cross_entropy_loss.py`, and `agrawal_accuracy_loss.py`
for some example usages.

Please help us in improving our work by contributing or pointing to issues. We will update this iteration soon with further information.

## Installation
Currently, no pip installation is possible. Please clone this repository and install the required packages as they are needed. 
`torch` for example is only required if torch models are to be explained (see the wrappers for this). Because of the new typing module this code will only work for rather new python versions. It is tested on python 3.9.7.
We will soon update this for better compatibility with existing code.
