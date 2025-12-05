# Entropy Production Rate Calculations for the Voter Model

This project includes two main methods to calculate entropy production rates (EPR) for the Voter Model, which is a spin-spin interaction-based model generally used in the context of opinion dynamics, chemical reactions, etc. The EPR is mainly used as a metric to delve deeper into understanding steady-state solutions for out-of-equilibrium systems. Our aim was to understand the reversibility/irreversibility properties of the Voter model.

## Transition Rates and Exact Calculations

For Voter Models with long-range interactions, we have two important parameters: the system size 'N' and the scaling parameter 'alpha'

For the theory see: [arxiv:2309.16517](https://arxiv.org/pdf/2309.16517)

'votermain.py' calculates EPR depending on 'N' and 'alpha' using exact transition matrix solving and the [Arnoldi Scheme](https://en.wikipedia.org/wiki/Arnoldi_iteration), the latter is an approximate method.

## Simulating the system to find EPR

The system produces entropy (positive or negative) every time it transitions between states, and the transition rates are different depending on the direction. 

'votersim.py' simulates the system as a continuous-time Markov process using the [Gillespie algorithm](https://en.wikipedia.org/wiki/Gillespie_algorithm) and calculates EPR from transitions.