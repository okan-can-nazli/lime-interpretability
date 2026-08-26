
---

**Task: Build a from-scratch LIME engine (NumPy only) for interpreting from-scratch LSTM forecasting models**

**Context:** Two from-scratch LSTM projects exist — weather LSTM (30-day window → next-day tavg) and Caputo fractional-derivative LSTM (COVID case forecasting). Both pure NumPy, no ML libraries. This task adds interpretability via LIME, built the same way: from scratch, no `lime`/`sklearn`/`scipy`.

**Goal:** Given one specific prediction from either model, explain which input features (which day, which variable) drove that prediction — by fitting a local weighted linear surrogate around perturbed versions of that input.

**Architecture — package, not single file, one class per file:**

```
lime/
    __init__.py
    sampler.py      -> Sampler class
    kernel.py        -> Kernel class
    surrogate.py       -> LocalSurrogate class
    explainer.py       -> LimeExplainer class (orchestrator)
```

Classes are loosely coupled: each `__init__` takes the *output* of the previous stage as a plain parameter (e.g. `Kernel(dist_sq_perturbations, sigma)`), never the previous class instance itself. No class reaches into another's internals via shared state.

1. **`sampler.py` — `Sampler`** ✅ DONE
   - Generates `n` perturbed neighbours of one input instance `x` within radius `r`, using zero-mean Gaussian noise (`np.random.randn`, not `np.random.rand` — must be centered, not one-sided)
   - Noise scale auto-clamped via `min(noise_frac, r / np.sqrt(x.size))` to keep rejection-sampling likely to succeed
   - Guarded against infinite loop with `max_tries` + explicit `raise ValueError`
   - Stores `self.perturbations` (accepted neighbours) and `self.dist_sq_perturbations` (matching squared distances, same index correspondence — no `continue`-related off-by-one bug)

2. **`kernel.py` — `Kernel`** 🔧 IN PROGRESS
   - Input: `dist_sq_perturbations` (from Sampler output, passed as plain list/array — not the Sampler object), `sigma` (kernel width hyperparameter)
   - Computes proximity weight per neighbour: `w_i = exp(-d_i^2 / sigma^2)`
   - Output: `self.weights`, vectorized via NumPy (no manual Python loop)
   - Role in pipeline: turns "how far is this neighbour" into "how much should the regression trust this neighbour" — closer neighbours get weight near 1, far ones near 0, which is what makes step 3 a *local* fit rather than a global one

3. **`surrogate.py` — `LocalSurrogate`** ⬜ NOT STARTED
   - Closed-form weighted ridge regression, `β = (XᵀWX + λI)⁻¹XᵀWy`, plain NumPy (`np.linalg.solve`)
   - `X`: perturbations (or their interpretable/mask representation — TBD), `W`: diag(weights from Kernel), `y`: model predictions on perturbations, `λ`: ridge regularizer

4. **`explainer.py` — `LimeExplainer`** ⬜ NOT STARTED
   - Orchestrates Sampler → Kernel → LocalSurrogate given a `predict_fn` and raw instance
   - Knows nothing about LSTMs, gates, or either specific dataset

5. **`adapters/`** ⬜ NOT STARTED
   - Task-specific, thin wrappers, one per project
   - Wraps trained model forward pass into flat `predict_fn(flattened_input) -> scalar`
   - Defines what a "feature" is per domain and what a sensible perturbation baseline is

**Constraints:**
- No external ML libraries anywhere — NumPy only
- Engine reusable unchanged across both projects; only adapters differ
- File naming: plain ASCII only (`lime_engine.py`/`lime/` package — avoid non-ASCII characters like `İ` in filenames, can break imports depending on filesystem)

**Not in scope yet:** LIME variants (K-LIME, Anchor, etc.)

---

