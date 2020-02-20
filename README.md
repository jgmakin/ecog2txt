# ecog2txt
Code for decoding speech as text from neural data

This package contains Python code for the high-level aspects of decoding speech from neural data, including transfer learning across multiple subjects.  It was used to for all results in the paper "Machine translation of cortical activity to text with an encoder-decoder framework" (Makin et al., _Nature Neuroscience_, 2020).  These high-level aspects include the structuring of the training, the organization by subjects, and the construction of [tf_records](https://www.tensorflow.org/tutorials/load_data/tfrecord).  The training itself is done with the adjacent [`machine_learning` package](https://github.com/jgmakin/machine_learning), which implements sequence-to-sequence networks in [TensorFlow](https://www.tensorflow.org).

## Installation
...

## Getting started
In order to unify the vast set of parameters (paths, experimental block structure, neural-network hyperparameters, etc.), all experiments are organized with the help of two configuration files, `block_breakdowns.json`, and `YOUR_EXPERIMENT_manifest.yaml`.  Examples of each are included in this repository.

1.  Edit the `block_breakdowns.json` to match your use case.  The entries are
  ```
  SUBJECT_ID: {BLOCK: {"type: BLOCK_TYPE, "default_dataset": DEFAULT_DATASET_VALUE}}`
  ```
where the `DEFAULT_DATASET_VALUE` is one of `"training"/"validation"/"testing"`; and the `BLOCK_TYPE` is whatever descriptive title you want to give to your block (e.g., `"mocha-3"`).  Assigning types to the blocks allows them to be filtered out of datasets, according to information provided in the `experiment_manifest.yaml` (see next item).

Place your edited copy into a directory we will call `json_dir`.

2.  Edit `example_experiment_manifest.yaml` to something sensible for your case.  The *most important thing to know* is that many of the classes in this package (and `machine_learning`) load their default attributes from this `manifest`.  That means that, even though the keyword arguments for their constructors (`__init__()` methods) may appear to default to `None`, this `None` actually instructs the class to default to the argument's value in the `manifest`.

    You don't have to set all the values before your first run, but in the very least, you should:
    * Fix the paths/dirs.  For the most part they are for writing, not reading, so you can set them wherever you like.  For the three reading paths:
      * `json_dir` must point to the location of your `block_breakdowns.json` file (see previous item).
      * `channels_path` must point to a (possibly empty) plain-text file listing (one entry per line) any bad channels.  NB that these are assumed to be 1-indexed!
      * `electrode_path`: you can ignore this unless you plan to plot results on the cortical surface (in which case contact me).
    * Additionally, if you want to allow blocks of a certain type (see previous item) to appear in one of the `training/validation/testing` sets, you need to fill in the `block_types` entry accordingly.  (For example, in the `example_experiment_manifest.yaml`, the `testing` and `validation` sets are allowed to include only `mocha-1`, but the training set is allowed to include `mocha-1, ..., mocha-9`.)
    * `grid_size`: set this to match the dimensions of your ECoG grid
    * `vocab_file`: 
    * `DataGenerator`: see below.

    You can probably get away with leaving the rest of the values in the `.yaml` at their default values, at least for your first run.
