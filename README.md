# SwapQMC
[![Build Status](https://github.com/TongSericus/SwapQMC/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/TongSericus/SwapQMC/actions/workflows/CI.yml?query=branch%3Amain)

Auxiliary-field quantum Monte Carlo (AFQMC) implementation of swap algorithm for measuring entanglement entropy and other QI-related quantities in interacting systems. Current version only supports Hubbard-type model but is extendable to general systems that are sign-problem-free.

To run the code, several example scripts are provided in the [examples](https://github.com/TongSericus/SwapQMC/tree/main/examples) folder.

The module itself does not provide any built-in measurement functions expect for entanglement-related measures. Nevertheless, regular measurements can be performed on the script level. Example scripts on measuring spin correlation functions can be found in [examples/utils](https://github.com/TongSericus/SwapQMC/tree/main/examples/utils) folder.