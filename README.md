# From-Scratch LIME Engine (NumPy only)

A LIME-style interpretability engine built entirely from scratch in NumPy — no `lime`, no `sklearn`, no `scipy` — used to explain individual predictions of the from-scratch LSTM forecasting models (weather LSTM: 30-day window → next-day `tavg`, and the Caputo fractional-derivative LSTM: COVID-19 case forecasting).

Given one specific prediction from either model, the engine explains which input features (which day, which variable) drove that prediction, by fitting a local, weighted linear surrogate around perturbed versions of the input.

## Approach

For a given instance `x` and model `predict_fn`, the engine:

1. Samples perturbed neighbours of `x`
2. Weights each neighbour by proximity (closer neighbours matter more)
3. Fits a local weighted linear surrogate on the (perturbation, prediction) pairs
4. Reads off the surrogate's coefficients as per-feature attributions

## Package layout

```
lime/
    __init__.py
    sampler.py      -> Sampler class
    kernel.py       -> Kernel class
    surrogate.py    -> LocalSurrogate class
    explainer.py    -> LimeExplainer class (orchestrator)
adapters/
    weather_adapter.py   -> wraps the weather LSTM's forward pass into predict_fn
    covid_adapter.py     -> wraps the Caputo fractional LSTM's forward pass into predict_fn
```

Classes are loosely coupled: each `__init__` takes the *output* of the previous stage as a plain parameter (e.g. `Kernel(dist_sq_perturbations, sigma)`), never the previous class instance itself. No class reaches into another's internals via shared state.

## Components

### `sampler.py` — `Sampler`
- Generates `n` perturbed neighbours of one input instance `x` within radius `r`, using zero-mean Gaussian noise (`np.random.randn`, centered rather than one-sided)
- Noise scale auto-clamped via `min(noise_frac, r / np.sqrt(x.size))` to keep rejection-sampling likely to succeed
- Guarded against infinite loops with `max_tries` + explicit `raise ValueError`
- Stores `self.perturbations` (accepted neighbours) and `self.dist_sq_perturbations` (matching squared distances, same index correspondence)

### `kernel.py` — `Kernel`
- Input: `dist_sq_perturbations` (from `Sampler`, passed as a plain array — not the `Sampler` object itself), `sigma` (kernel width hyperparameter)
- Computes a proximity weight per neighbour: `w_i = exp(-d_i^2 / sigma^2)`
- Output: `self.weights`, fully vectorized via NumPy
- Role in the pipeline: turns "how far is this neighbour" into "how much should the regression trust this neighbour" — closer neighbours get weight near 1, far ones near 0, which is what makes the surrogate fit *local* rather than global

### `surrogate.py` — `LocalSurrogate`
- Closed-form weighted ridge regression, `β = (XᵀWX + λI)⁻¹XᵀWy`, solved via `np.linalg.solve`
- `X`: perturbations (interpretable representation), `W`: `diag(weights)` from `Kernel`, `y`: model predictions on the perturbations, `λ`: ridge regularizer

### `explainer.py` — `LimeExplainer`
- Orchestrates `Sampler → Kernel → LocalSurrogate` given a `predict_fn` and a raw instance
- Knows nothing about LSTMs, gates, or either specific dataset — fully model-agnostic

### `adapters/`
- Task-specific, thin wrappers, one per project
- Wraps the trained model's forward pass into a flat `predict_fn(flattened_input) -> scalar`
- Defines what a "feature" is per domain (lag position / variable) and what a sensible perturbation baseline is

## Usage

```python
from lime.explainer import LimeExplainer
from adapters.covid_adapter import predict_fn, instance

explainer = LimeExplainer(predict_fn=predict_fn, r=0.5, n=500, sigma=0.25, lam=1.0)
attribution = explainer.explain(instance)
# attribution: per-feature (lag, variable) weights for this one prediction
```

## Constraints

- No external ML libraries anywhere — NumPy only
- Engine is reusable unchanged across both projects; only the adapters differ
- File naming: plain ASCII only (avoids filesystem issues with non-ASCII characters in filenames)

## Not in scope

- LIME variants (K-LIME, Anchor, etc.)
