# Weka S3 Gateway Benchmarking (USDF to IDF)

Adjust the tests run by using different items from the `potential_datasets` dict in the third cell.

Add new datasets there on both the `main` and `mobu` branches.

Use `main` for development, data analysis, and for running individual tests.
[Speed-plot.ipynb](./Speed-plot.ipynb) only exists there.

Use the `mobu` branch for automated scale testing.
Flock instructions are in [README.md](./README.md) on the `mobu` branch.

If you modify [weka_helpers.py](./weka_helpers.py) on the `main` branch, it is your responsibility to update the corresponding Python on the `mobu` branch.
You should, generally, be editing on `main`, and should copy the content of [weka_helpers.py](./weka_helpers.py) into a paste buffer, and then paste it at the bottom of the first cell on the `mobu` branch, ensuring that the imports in that cell are commented out.
If you find yourself developing on `mobu`, you are doing it wrong.

The reason the branches differ is that under `mobu`, cell execution happens by executing a (possibly very long) string representing a notebook's JSON.
It has no concept of a checkout, or even, really, of the file it's executing, so it is necessary to put everything into a single notebook.
