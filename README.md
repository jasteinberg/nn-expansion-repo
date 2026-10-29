# nn-expansion: circuit expansion for learning in neural networks

Code that reproduces every data figure of

> J. Steinberg, M. Advani and H. Sompolinsky, *New role for circuit expansion for
> learning in neural networks*, Physical Review E **103**, 022404 (2021),
> Editors' Suggestion. https://doi.org/10.1103/PhysRevE.103.022404
> Preprint: https://arxiv.org/abs/2008.08653

The implementation is written from the equations and appendices of the paper.

<small>*Drafted with the assistance of Claude (Anthropic).*</small>

## The model in brief

A student perceptron learns from a teacher perceptron with weights $w^0 \in \mathbb{R}^{N_0}$ labels inputs $x_i \sim \mathcal N(0,1)$ through the
noisy field

$$h_0 = \frac{1}{\sqrt{N_0}}\, w^0\cdot x + \epsilon, \qquad \epsilon \sim \mathcal N(0, \sigma_{\rm out}^2), \qquad y_0 = \mathrm{sgn}(h_0).$$

Due to label noise from the teacher, training examples will no longer be linearly separable i.e., perfectly classified by a perceptron,
beyond some critical number of examples, rendering the training task as ``unrealizable" by a perceptron. To increase the capacity, the student is widened to
$N = \beta N_0$ inputs by appending $N_+ = N - N_0$ expanded units $z$, and its field is
$h = (w\cdot x + \tilde w \cdot z)/\sqrt N$. Two expansions are studied. In the deterministic
expansion the units are a fixed random threshold layer,

$$z_j = \frac{A}{\sqrt{f(1-f)}}\Big[\Theta\Big(\sum_i J_{ji} x_i - T\Big) - f\Big],$$

with sparsity $f$. In the stochastic expansion they are drawn afresh for every
pattern, $z_j \sim \mathcal N(0, \sigma_{\rm in}^2)$, independent of $x$. The weights are learned by max-margin
classification without bias,

$$\min_{w,\tilde w}\ |w|^2 + |\tilde w|^2 \quad \text{s.t.}\quad y_0^\mu h^\mu \ge 1 \ \ \forall \mu ,$$

and the expanded weights $\tilde w$ are pruned after learning. The generalization error
then depends only on the overlap $R$ between $w$ and $w^0$,

$$E_g(R) = \frac{1}{\pi}\Big[\frac{\pi}{2} - \arctan\frac{R}{\sqrt{1 + \sigma_{\rm out}^2 - R^2}}\Big].$$

Expansion raises the capacity, $\alpha_{0,c}(\beta) = \beta\,\alpha_c(\sigma_{\rm out})$ at $\sigma_{\rm in} = 1$, so the student can train on
more examples. The central result is that this improves generalization after
pruning, most of all for sparse expansions. For the stochastic expansion a
replica mean-field theory, exact as $N \to \infty$, gives $E_g$ in closed form, and the
expansion is equivalent to an unexpanded network trained with slack
regularization (Sec. III C). The paper extends the result to a quadratic teacher
and a quadratic-kernel student, and compares with logistic regression trained
by gradient descent.

## Installation

The base install needs only numpy, scipy, matplotlib and joblib.

```bash
pip install -e .
pip install -e ".[dev]"    # adds pytest, mypy and ruff
python -m pytest
```

PyTorch is optional and is used only for GPU runs (see below): `pip install -e ".[gpu]"`.
The test suite (84 tests, about a minute) includes mypy and ruff.

## Reproducing the figures

```bash
python -m experiments.reproduce fig07 --profile quick --jobs 8
python -m experiments.reproduce all   --profile quick --jobs 8
```

Each command runs the simulations and the replica theory of a figure, saves the
results to `results/<figure>_<profile>.npz` with a `.json` record of the
configuration, git commit and run time, and draws every panel to
`results/<panel>_<profile>.png` with the published curve underneath in grey.
`--plot-only` redraws from the saved results.

Two profiles are defined for every figure in `configs/<figure>.toml`. `paper`
holds the parameters of the published figure. `quick` lists only the
parameters it changes (smaller $N_0$, fewer trials and grid points), which is
enough to see every effect. All quick runs together take a few minutes on 12
cores. Every key of the `paper` table is required and an unknown key raises an
error, so a typo cannot fall back to a default.

To draw single panels, for example in a notebook, use

```python
from experiments.notebook import list_panels, show
list_panels("quick")
show("fig08b", "quick")
```

which reads the saved results and never recomputes. `notebooks/nn_expansion.ipynb`
is a guided walkthrough of the model with small runs.

| Paper | Module (`experiments/`) | Agreement with the published figure |
|---|---|---|
| Figs. 2-4 | `fig02_04_deterministic` | simulation, same effects at quick scale |
| Fig. 5 | `fig05_capacity` | theory, relative error below $5\cdot10^{-5}$ |
| Fig. 6a, b | `fig06_theory` | theory, below $2\cdot10^{-4}$, same capacity cut-off |
| Fig. 6c, d | `fig06_theory` | theory, below $5\cdot10^{-5}$ |
| Fig. 7a | `fig07_sigma_in` | simulation, and theory below $10^{-4}$ |
| Fig. 7b | `fig07_sigma_in` | theory, $\sigma_{\rm in}^* = 0.3723$ as published |
| Fig. 8a, b | `fig08_stochastic_vs_deterministic` | simulation, Fig. 8b on the published curves at quick scale |
| Figs. 9, 10 | `fig09_10_quadratic_teacher` | simulation, same effects at quick scale |
| Fig. 11 | `fig11_kernel` | simulation, same effects at quick scale |
| Figs. 12-14 | `fig12_14_logistic` | simulation, Fig. 13 traces to about 0.003 |
| Sec. III C | `slack_correspondence` | check of Eqs. 25-30, not a published figure |

Fig. 1 is a schematic and is not reproduced. The theory agreements are tested
against the plotted data of the published figures, which are stored in
`reference/published/` (extracted from the published figure files with
`scripts/extract_published_figures.py`).

## Layout

```
nn_expansion/
  teacher.py      noisy linear, quadratic and kernel teachers; noise floor      Eqs. 1, 4, 31, 32
  expansion.py    deterministic and stochastic expansions; quadratic features    Eqs. 5, 6, 10
  svm.py          max margin without bias; L2-slack SVM                          Eqs. 8, 9, 25
  logistic.py     full-batch gradient descent on the logistic loss               Eqs. 38, 39
  theory.py       capacity, replica saddle points, large beta, optimal sigma_in  App. B-F
  simulate.py     one trial: teacher, data, expansion, learning, measurements
  sweep.py        planned sweeps of trials, seeds, shards, trial averaging
  backend.py      numpy (default) or torch for the learning rules
experiments/      one module per figure group: plan, compute, panels; reproduce.py is the CLI
configs/          paper and quick parameters of every figure
reference/        plotted data of the published figures
notebooks/        guided walkthrough
tests/            tests against closed forms, limits, invariances and the published data
```

## Trials, seeds and parallelism

Each figure module declares its simulations as series of trials before running
any of them. The seed of a trial is derived from the base seed, the series, the
grid point and the trial index, so results do not depend on `--jobs` or on the
order in which trials run.

**Sharding.** To split a run across machines (for example a SLURM array), run
`--shard k/n` for $k = 0,\dots,n-1$ at the same commit, then `--merge n`. Each
shard stores its trials under `results/shards/`. The merge refuses to finish if
a trial is missing or repeated, or if the shards were made from a different
configuration, and the merged result equals an unsharded run.

**GPU.** With PyTorch installed, `--device cuda` (or `cuda:k`) runs the
max-margin solver and gradient descent in float64 on an NVIDIA GPU. The default
`--device cpu` uses numpy. Training data are always drawn on the CPU, so a trial
sees the same data on every device. Each interior-point step of the solver forms
and factorizes the $N\times N$ matrix $I + G^\top D G$, which costs $O(PN^2)$, so the GPU pays off on
cards with fast float64 (for example A100 or H100) and not on consumer cards.
Apple `mps` is refused because it has no float64. Use one or a few workers per
GPU. The torch code path is tested on the CPU (`--device torch-cpu`).

## Implementation notes

**Max-margin solver.** The quadratic program is solved by a dense primal-dual
interior-point method in which each Newton step is one Cholesky factorization of
an $N\times N$ matrix. At $N = 1000$, $P = 4000$ a solve takes about 16 s on a 12-core laptop CPU. A
training set is declared non-separable only with a Farkas certificate: a convex
combination of the patterns $y^\mu \Phi^\mu$ whose norm vanishes, which by Gordan's theorem
excludes every separating $w$. Every returned solution is checked against the
KKT conditions ($\lambda \ge 0$, all margins $\ge 1$, zero duality gap), which certify the
optimum independently of the solver. The L2-slack SVM of Eq. 25 is the same
solver applied to features augmented by one private unit per pattern, which is
the construction of Eqs. 26-30.

**Replica theory at $\sigma_{\rm in} = 1$.** The integral of Eq. B14 has the closed form
$\int_{-k}^{\infty} Dx\,(k+x)^2 x = 2[\phi(k) + k\Phi(k)]$, so $m$ and $\kappa$ are explicit functions of
$k^* = \kappa/\sqrt{1-\bar m^2}$ and Eq. B13 becomes a single bracketed root in $k^*$. The bracket exists
exactly below capacity. The general-$\sigma_{\rm in}$ equations (B5-B11) are solved by continuation
in $\sigma_{\rm in}$ from the exact $\sigma_{\rm in} = 1$ solution.

**Generalization error.** Where the student is linear in $x$ and the teacher is the
noisy linear teacher, $E_g$ follows exactly from the trained weights through $E_g(R)$,
including the unpruned stochastic student, whose fresh test units are Gaussian.
Otherwise (deterministic unpruned, quadratic teacher, kernel student) $E_g$ is
estimated on fresh test inputs with fresh label noise. The teacher norms are
fixed, $|w^0|^2 = N_0$, so $\sigma_{\rm out}$ is the noise-to-signal ratio of every teacher.

**Trial averaging.** Near capacity some training sets are not separable. Each
figure uses the averaging rule of the published figure. For Fig. 7a a
single non-separable trial removes the point. For Figs. 2-4 and 8-11 a point is
removed when more than half of its trials fail, and otherwise averaged over the
separable trials. This conditions on separability, so near capacity it averages
over a selected subset of training sets. Error bars are one standard error for
Figs. 2-4, 8a and 12-14 and two for Figs. 7a, 8b and 9-11, as published. Both
choices are configuration entries.

**Runtime checks.** Besides the KKT certificate, every trial checks the shapes of
its inputs, that a max-margin solution separates its training set, that $R$ and
$E_g$ lie in their ranges, that the hidden layer is neither silent nor saturated,
and that gradient descent stays finite. A failed check raises an error.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
