# Weka S3 Gateway Benchmarking (USDF to IDF)

## Mobu Branch

Use the `mobu` branch to test under Mobu.

Use the following YAML for a Mobu flock.

It's currently present in Phalanx on branch `t/DM-56719`.

Adjust the count as desired.

Adjust the tests run by using different items from the `potential_datasets` dict in the third cell.

Add new datasets there on both the `main` and `mobu` branches.

```yaml
    - name: "weka-test"
      count: 50
      user_spec:
        username_prefix: "bot-mobu-weka"
        uid_start: 100500
        gid_start: 100500
      scopes:
        - "exec:notebook"
        - "exec:portal"
        - "read:image"
        - "read:tap"
      business:
        type: "NotebookRunnerCounting"
        options:
          cell_execution_timeout: "1h"
          repo_url: "https://github.com/lsst-sqre/weka-s3-benchmark.git"
          repo_ref: "mobu"
        restart: true
```

If you modify [weka_helpers.py](./weka_helpers.py) on the `main` branch, or the inlined helpers on the `mobu` branch, it is your responsibility to update the corresponding Python on the other branch.
If you modify [weka_helpers.py](./weka_helpers.py) on the `main` branch, it is your responsibility to update the corresponding Python on the `mobu` branch.
You should, generally, be editing on `main`, and should copy the content of [weka_helpers.py](./weka_helpers.py) into a paste buffer, and then paste it at the bottom of the first cell on the `mobu` branch, ensuring that the imports in that cell are commented out.
If you find yourself developing on `mobu`, you are doing it wrong.


The reason the branches differ is that under `mobu`, cell execution happens by executing a (possibly very long) string representing a notebook's JSON.
It has no concept of a checkout, or even, really, of the file it's executing, so it is necessary to put everything into a single notebook.
