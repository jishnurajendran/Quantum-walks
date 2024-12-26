# Quantum-walks



[![DOI](https://zenodo.org/badge/99251254.svg)](https://doi.org/10.5281/zenodo.14559714)

This code implements a quantum walk simulation with alternating coin operators, a quantum mechanical analog of the classical random walk. The simulation demonstrates the unique quantum interference patterns that emerge from using two different coin operators (A and B) alternately at each step. The walker starts at the origin and evolves through a superposition of states, with its movement controlled by the alternating quantum coin flips.

The coin operators are parameterized by angles ($\alpha$, $\beta$, $\gamma$) which determine the quantum rotation, allowing for exploration of different walking behaviors. The code calculates and visualizes both the final probability distribution of the walker's position and the step-by-step evolution of left-right probability differences ($P_L$ - $P_R$). This implementation is particularly useful for studying quantum transport phenomena and quantum interference effects.

Key features:
- Vectorized calculations for improved performance
- Customizable coin operators through angle parameters
- Visualization of probability distributions and asymmetry measures
- Statistical analysis of the quantum walk outcomes

The simulation demonstrates the non-classical behavior of quantum walks, particularly how the alternating coins can create unique interference patterns different from single-coin quantum walks or classical random walks.

This code and an improved version is used in article: 
- Implementing Parrondo’s paradox with two-coin quantum walks
- Playing a true Parrondo's game with a three-state coin on a quantum walk