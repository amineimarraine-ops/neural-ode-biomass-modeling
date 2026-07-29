# Neural ODE Biomass Modeling

This project explores the use of **Neural Ordinary Differential Equations (Neural ODEs)** to model biological dynamical systems.

The main goal is to evaluate the ability of Neural ODEs to learn **metabolic dynamics from experimental data** using the [`jax`](https://github.com/jax-ml/jax) library.

## Project Status

This section summarizes what has been completed, what has not yet been completed, and what is currently in progress.

## What Has Been Done

- Training using **Run1 only**. The results obtained with Run1 were unsatisfactory.
- Training using **Run2 only**, since Run1 did not produce satisfactory results.
- Exclusion of bioreactors **13** and **19**, as they exhibited abnormal biomass profiles that were inconsistent with the expected biological behavior. These bioreactors were therefore considered experimental artifacts or failed experiments.
- Leave-one-bioreactor-out cross-validation performed on **Run2**:
  - One bioreactor is excluded from the training set.
  - The model is trained on the remaining bioreactors.
  - The excluded bioreactor is then used as the test bioreactor.
- Data normalization is performed using **training-data statistics only** to prevent data leakage from the test bioreactor.

### Cross-Validation Results

![Image description](results.png)

## What Has Not Yet Been Done

- Training on **Run1 and Run2 simultaneously**.
- Full implementation of the alternative extrapolation benchmark designed to answer the following question:

> How much training data is required to learn dynamics that generalize to unseen bioreactors?

Some preliminary experiments have already been performed, but the results were not particularly promising.

Since the main extrapolation pipeline is still being implemented, this alternative benchmark has not been investigated further at this stage. Besides, this strategy does not answer the question that was raised in the first place.

## Work in Progress

A new extrapolation benchmark is currently being implemented.

The model is trained normally on the training bioreactors. One bioreactor is completely excluded from training and used as the test bioreactor.

For the test bioreactor:

1. The trajectory is observed only up to a selected cutoff time.
2. The measured state at the cutoff time is used as the initial condition of the Neural ODE.
3. The model is integrated forward from this state.
4. The predicted trajectory is compared with the remaining experimental trajectory.

This benchmark evaluates the model's ability to forecast the future evolution of an unseen bioreactor from a partially observed trajectory.

This differs from the previous experiment, which evaluated how much training data was required to learn dynamics that generalize to unseen bioreactors.

The implementation and validation of this extrapolation pipeline are currently in progress.

### Recent Extrapolation Results

<!-- Insert figures, tables, or result summaries below. -->

![Recent extrapolation results](images/extrapolation_results.png)
