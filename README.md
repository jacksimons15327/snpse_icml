# (S)NPSE

Repository for the paper "Sequential Neural Score Estimation: Likelihood-Free Inference with Conditional Score-Based Diffusion Models," published at ICML 2024.

[TO DO: include link to published paper.]

## Notes

- **Environment Setup:** The environment can be set up using the `requirements.txt` file provided.
- **Compatibility:** The code should work on both CPU and GPU.
- **Usage:**
  - `main.py` is for running individual results locally.
  - `main_array_local.py` is for running array jobs locally.
  - `main_array.py` is for running array jobs on Slurm.
- **Configuration:** Default configurations are located in `config.py`.
  - Note that the optimal configurations for sequential and non-sequential runs will differ.
  - `config.score_network.use_energy` refers to parameterising the score-network as an energy model (see Appendix G of the paper). This greatly speeds up the truncated proposal prior.
  - `config.score_network.t_sample_size` refers to the number of sweeps of the entire dataset before early stopping criteria are computed. Empirically, `~10` performs well, though this will slow down the code.

## Installation

To set up the environment, run:
```sh
conda create --name snpse --file requirements.txt
conda activate snpse
```