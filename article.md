# Post-training a 7B vision-language model to use a computer

![The coding loop after training: open Jupyter, read a NameError, and rerun a passing fix](examples/banner.jpg)

I post-trained [Qwen2.5-VL-7B](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) to complete a desktop coding loop: open Jupyter, write or edit code, run it, read the output, and apply a fix when needed. The untrained model scored 0/24 on a held-out test set.

This write-up covers the task, the evaluation protocol, the training stages, and the measured results.

## Task

The model receives a screenshot and outputs UI actions such as clicks and keystrokes. Each episode follows the same loop. The stills and clips below are from one held-out NameError task (`repair_has_digit`) after training.

1. Open Jupyter

JupyterLab after the model opens `repair_has_digit.ipynb`.

![Open Jupyter](examples/steps/01-open-jupyter.jpg)

Clip: [01-open-jupyter.mp4](examples/steps/01-open-jupyter.mp4)

2. Write or edit code

The provided cell. The print call uses `has_digi` instead of `has_digit`.

![Write or edit code](examples/steps/02-edit-code.jpg)

Clip: [02-edit-code.mp4](examples/steps/02-edit-code.mp4)

3. Run it

The first execution of that cell (execution count `[1]`).

![Run the cell](examples/steps/03-run.jpg)

Clip: [03-run.mp4](examples/steps/03-run.mp4)

4. Read the output or traceback

`NameError: name 'has_digi' is not defined`.

![Read the traceback](examples/steps/04-read-traceback.jpg)

Clip: [04-read-traceback.mp4](examples/steps/04-read-traceback.mp4)

5. Apply a fix if needed

The print call corrected to `has_digit`.

![Apply a fix](examples/steps/05-apply-fix.jpg)

Clip: [05-apply-fix.mp4](examples/steps/05-apply-fix.mp4)

6. Run it again

The second run prints `True`.

![Run again](examples/steps/06-run-again.jpg)

Clip: [06-run-again.mp4](examples/steps/06-run-again.mp4)

The tasks included writing a new function, debugging a provided file, handling a changed window layout, and fixing a `.py` file from the terminal only. I used Jupyter because that is the environment I could run reliably.

## Evaluation

I froze two test sets before training and did not use them as training data:

- **Original 24.** Write a new function, debug a provided file, handle a changed layout, or use the terminal only.
- **Debug 35.** Five tasks each for seven bug types: syntax, indentation, name errors, wrong operator, off-by-one, wrong comparison, and a Python file fixed from the terminal.

A programmatic checker inspects files on disk (for example, whether a function exists or a script prints the expected output). If the model reports "done" and the checker fails, the episode is counted as a fail. I also track those cases separately as false "done" claims.

For trained checkpoints I report the mean of 3 runs at temperature 0.2. The untrained score of 0/24 is from an earlier single run on the same original-24 set.

The frozen tests can be checked with:

```bash
cd tests/original-24 && sha256sum -c FROZEN.sha256
cd ../debug-35 && sha256sum -c FROZEN.sha256
```

## Supervised fine-tuning

I recorded successful demonstrations of the loop and fine-tuned the model on those traces.

On the original 24 tasks, the first fine-tune averaged 17.0/24 (71%). After I added more careful debugging demonstrations, it averaged 21.0/24 (88%).

On the debug set, the first fine-tune averaged 4.7/35 (13%), with about 30 false "done" claims per run. Typical failures were editing the wrong line, rewriting a test, applying a fix without rerunning the file, or reporting done after seeing an error.

Here is the end of one of those failures, a NameError the early checkpoint did not fix:

![Before training, the model fails a NameError task](examples/before.gif)

I then recorded demonstrations with a narrower edit policy: change only the buggy line, run the file, read the output, then stop. Debug-set accuracy went from 13% to 61% (21.3/35). False "done" claims dropped from 30.0 to 13.7.

![Debugging scores by bug type](results/debug_by_type.png)

Improvement was uneven across bug types after those demonstrations.

## Reinforcement learning

I ran one round of GRPO with group size 8. Trajectories that passed the checker were upweighted. Failures and early quits were downweighted. A penalty term limited drift from the fine-tuned model.

After that round, debug-set accuracy went from 61% to 70% (24.3/35). False "done" claims fell from 13.7 to 10.7. Most of the gain was on debugging tasks, particularly name errors.

Here is a NameError episode after that training:

![After training, the model fixes a NameError task](examples/after.gif)

A sped-up comparison of the full attempts is in [`examples/before-after.mp4`](examples/before-after.mp4).

On the original 24 tasks, the RL checkpoint averaged 20.7/24 (86%). The second supervised run averaged 21.0/24 (88%).

A second RL round matched the first round on both test sets. I stopped after that.

![Overall scores](results/overall.png)

| Model | Original 24 | Debug 35 |
| --- | --- | --- |
| After early fine-tuning | 17.0 / 24 (71%) | 4.7 / 35 (13%) |
| After more targeted demos | **21.0 / 24 (88%)** | 21.3 / 35 (61%) |
| After one round of RL | 20.7 / 24 (86%) | **24.3 / 35 (70%)** |

The code, frozen tests, and charts are in [computer-use-post-training](https://github.com/SarnadAbhilash/computer-use-post-training).
