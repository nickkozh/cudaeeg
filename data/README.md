# data/ — EEG drop folder

Drop a recording into this directory and the pipeline (and visualizers) will pick it up.

## Supported formats

| Extension | Loader | Layout expected | Sample rate |
|---|---|---|---|
| `.csv` | `numpy.loadtxt` | rows = samples, cols = channels (auto-transposed if needed) | assumes 250 Hz |
| `.edf` | `pyEDFlib.highlevel.read_edf` | as encoded; `fs` read from header | per file |
| `.npy` | `numpy.load` | `(n_ch, n_samp)` — auto-transposed if rows > cols | assumes 250 Hz |

## Validation

- File must have at least the requested number of channels (`--channels` CLI flag). If more, the first N channels are used.
- Sample rate is taken from the EDF header; CSV/NPY are assumed to be 250 Hz. Mismatches log a warning but processing proceeds.
- The newest file (by mtime) is used when more than one is present.

## Behavior when empty

If `data/` is empty (or you don't pass `--file`), the pipeline falls back to synthetic EEG: pink noise + 10 Hz alpha on channels 0–1 + 60 Hz mains. Useful for demos and CI.

## Watching for new files

`data_loader.FolderWatcher` polls this directory at 1 Hz. New files are picked up at the next call to `load_windows(...)`; in-flight windows are not preempted.
