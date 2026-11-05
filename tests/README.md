# Locked test sets

These two sets were frozen before the final training stage. None of these tasks were used as training data.

## original-24

24 tasks in four tiers:

- **T1** — write a new function in Jupyter
- **T2** — debug a buggy notebook or file
- **T3** — same kind of work, but the screen layout is changed
- **T4** — fix a `.py` file using the terminal only

## debug-35

35 debugging tasks: 5 tasks for each of 7 bug types (syntax, indentation, name errors, wrong operator, off-by-one, wrong comparison, and a `.py` file fixed from the terminal).

## Check that nothing changed

```bash
cd original-24 && sha256sum -c FROZEN.sha256
cd ../debug-35 && sha256sum -c FROZEN.sha256
```

Both commands should print `OK` for every file.
