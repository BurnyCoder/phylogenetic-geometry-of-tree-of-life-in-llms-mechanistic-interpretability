# Learnings: experiment 04 (stage 3, behaviour and causal tests)

- **Check a rank test's p-value floor before running it.**
  - *What went wrong.* With n random controls, a rank p-value cannot go below 1/(n+1). The first run used 20 and 10 random subspaces, so its floors (0.048, 0.091) were above the Holm threshold of 0.025: the registered test could never pass.
  - *How it was found.* The first run returned both p-values exactly at their floors.
  - *Fix.* 99 subspaces, giving a floor of 0.01, fixed in deviation 19 before the rerun.
- **Variance-match on the set you sample from.**
  - *What went wrong.* The first run computed the probe's captured variance over all species but drew the random subspaces on the training species, so they were under-matched.
  - *Effect.* After matching on the training species, the mean random ablation drop rose and the ablation rank test became non-significant. An under-matched control flatters the method.
- **The mean of the null is not the null.**
  - *What happened.* A per-species Wilcoxon test against the *mean* random drop was very significant, while about one random subspace in nine matched the probe.
  - *Why the rank test is primary.* It uses the whole null distribution across subspaces, so it was registered as primary (deviation 19). The Wilcoxon test is kept as secondary.
- **A causal test inherits the confounds of its probe and readout.**
  - *The confound.* The probe is trained on all-pairs path length, which the 13 coarse groups dominate, and the readout is the class word. A positive C-swap is therefore expected from class information alone.
  - *The test.* Adding a class-only subspace (deviation 20) turned an apparent causal confirmation into "used for class, not shown for phylogeny".
  - *Lesson.* Pair every causal test with a control subspace that carries only the confounding information.
- **Compute baseline and edited losses with one function.** The first run's off-target losses used different sentence sets and weighting. A unit test now checks that an identity edit reproduces the baseline loss exactly.
- **Environment: a running process can lock the editable install on Windows.**
  - *What happened.* `uv run` tried to re-sync while another Python process (the trackio dashboard) held the venv. It failed halfway and removed the editable-install `.pth` file.
  - *Workaround.* Run with `PYTHONPATH=src` and `.venv/Scripts/python.exe` until the lock is released. Alternatively, `uv run --no-sync` avoids the rebuild when the install is intact.
