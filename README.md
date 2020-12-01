# ecog2txt
Code for decoding speech as text from neural data

This package contains Python code for the high-level aspects of decoding speech from neural data, including transfer learning across multiple subjects.  It was used for all results in the paper "Machine translation of cortical activity to text with an encoder-decoder framework" (Makin et al., _Nature Neuroscience_, 2020).  These high-level aspects include the structuring of the training, the organization by subjects, and the construction of [`TFRecord`](https://www.tensorflow.org/tutorials/load_data/tfrecord)s.  The (low-level) training itself is done with the adjacent [`machine_learning` package](https://github.com/jgmakin/machine_learning), which implements sequence-to-sequence networks in [TensorFlow](https://www.tensorflow.org).

## Installation
1.  Install [TensorFlow 1.15.3](https://www.tensorflow.org), the final version of TF1.x.
    ```
    pip install tensorflow-gpu==1.15.3
    ```
    If you don't have a GPU you should install the CPU version
    ```
    pip install tensorflow==1.15.3
    ```
    Please consult the Tensorflow installation documents.  The most important facts to know are that TF1.15 requires CUDA 10.0, `libcudnn7>=7.6.5.32-1+cuda10.0`, and `libnccl2>=2.6.4-1+cuda10.0`.  (I have only tested with up to, not beyond, the listed versions of these libraries).  Make sure the driver for your GPU is compatible with these versions of the cudNN and NCCL libraries.

2.  Install the three required packages:
    ```
    git clone https://github.com/jgmakin/utils_jgm.git
    pip install -e utils_jgm

    git clone https://github.com/jgmakin/machine_learning.git
    pip install -e machine_learning

    git clone https://github.com/jgmakin/ecog2txt.git
    pip install -e ecog2txt

    ```

## Getting started
In order to unify the vast set of parameters (paths, experimental block structure, neural-network hyperparameters, etc.), all experiments are organized with the help of two configuration files, `block_breakdowns.json`, and `YOUR_EXPERIMENT_manifest.yaml`.  Examples of each are included in this repository.

1.  Edit the `block_breakdowns.json` to match your use case.  The entries are

    ```SUBJECT_ID: {BLOCK: {"type: BLOCK_TYPE, "default_dataset": DEFAULT_DATASET_VALUE}}```
    
    where the `DEFAULT_DATASET_VALUE` is one of `"training"`/`"validation"`/`"testing"`; and the `BLOCK_TYPE` is whatever descriptive title you want to give to your block (e.g., `"mocha-3"`).  Assigning types to the blocks allows them to be filtered out of datasets, according to information provided in the `experiment_manifest.yaml` (see next item).
    Place your edited copy into a directory we will call `json_dir`.

2.  Edit `example_experiment_manifest.yaml` to something sensible for your case.  The *most important thing to know* is that many of the classes in this package (and `machine_learning`) load their default attributes from this `manifest`.  That means that, even though the keyword arguments for their constructors (`__init__()` methods) may appear to default to `None`, this `None` actually instructs the class to default to the argument's value in the `manifest`.

    You don't have to set all the values before your first run, but in the very least, you should:
    * Fix the paths/dirs.  For the most part they are for writing, not reading, so you can set them wherever you like.  For the three reading paths:
      * `json_dir` must point to the location of your `block_breakdowns.json` file (see previous item).
      * `bad_electrodes_path` must point to a (possibly empty) plain-text file listing (one entry per line) any bad channels.  NB that these are assumed to be 1-indexed! (but will internally be converted to zero-indexing).  Alternatively, you can provide (either via the manifest or as an argument to the `ECoGDataGenerator`) the `good_electrodes` directly.
      * `electrode_path`: you can ignore this unless you plan to plot results on the cortical surface (in which case contact me).
    * `block_types`: these set *necessary* conditions for membership in one of the datasets, `training`/`validation`/`testing`.  For example, in the `example_experiment_manifest.yaml`, the `testing` and `validation` sets are allowed to include only `mocha-1`, but the training set is allowed to include `mocha-1, ..., mocha-9`.  So if a `mocha-3` block has `validation` as its `"default_dataset"` in the `block_breakdowns.json`, it would be excluded altogether.
    * `grid_size`: Set this to match the dimensions of your ECoG grid.
    * `text_sequence_vocab_file`: You can provide a file with a list, one word per line, of all words to be targeted by the decoder.  This key specifies just the *name* of the file; the file itself must live in the `text_dir` specified in `__init__.py`.  If you set this key to `None`, the package will attempt to build a list of unique targets directly from the `TFRecord`s.  An example vocab_file, `vocab.mocha-timit.1806`, is included in this package.
    * `data_mapping`: Use this to set which data to use as inputs and outputs for the sequence-to-sequence network--see `_ecog_token_generator` below.  
    * `DataGenerator`: In the `example_experiment_manifest.yaml`, this points to the `ECoGDataGenerator` in `data_generators.py`, but you will probably want to subclass this class and point to your new (sub)class instead--see next item.

    You can probably get away with leaving the rest of the values in the `.yaml` at their default values, at least for your first run.
    
    Finally, make sure your `experiment_manifest.yaml` lives at the `text_dir` specified in `__init__.py` (you can change this as you like, but remember that the `text_sequence_vocab_file` must live in the same directory).

3. `ECoGDataGenerator`, found in `data_generators.py`, is a shell class for generating data--in more particularly for writing out the `TFRecords` that will be used for training and assessing your model--that plays nicely with the other classes.  However, three of its (required!) methods are unspecified because they depend on how *you* store *your* data.  (Dummy versions appear in `ECoGDataGenerator`; you can inspect their input and outputs there.)  You should subclass `ECoGDataGenerator` and fill in these methods:
    * `_ecog_token_generator`: a Python generator that yields data structures in the form of a `dict`, each entry of which corresponds to a set of inputs and outputs on a single trial.  For example, the entries might be `ecog_sequence`,`text_sequence`, `audio_sequence`, and `phoneme_sequence`.  The last two are not strictly necessary for speech decoding and can be left out--or you can add more.  Just *make sure that you return at least the data structures requested in the `data_mapping` specified in the `manifest`*.  So e.g. if the `data_mapping` is
    ```data_mapping = {'decoder_targets': 'text_sequence', 'encoder_inputs': 'ecog_sequence'}```
    then `_ecog_token_generator` must yield dictionaries containing *at least* (but not limited to) a `text_sequence` and an `ecog_sequence`.  The entire dictionary will be written to a `TFRecord` (one for each block), so it's better to yield more rather than fewer data structures, in case you change your mind later about the `data_mapping` but don't want to have to rewrite all the `TFRecord`s.
    
       And one more thing: the `text_sequence_vocab_file` key in the experiment manifest is linked to the `text_sequence` in this data mapping.  So if you plan to call your `decoder_targets` something else, say `my_words`, then make sure to rename the key in the experiment manifest that points to a vocab file to `my_words_vocab_file`.
    * `_get_wav_data`: should return the `sampling_rate` and audio `signal` for one (e.g.) block of audio data.  This will allow you to make use of the built-in `_get_MFCC_features` in constructing your `_ecog_token_generator`.  If you're never going to generate an `audio_sequence`, however, you can ignore it.
    * `_query`: should return the total number of examples in a group of blocks.  This will allow you to allocate memory efficiently when using the `get` method.  However, the methods `_query` and `get` are not used elsewhere in the code; they are convenience functions for examining the data directly rather than through a `TFRecord`.
    

## Training a model
The basic commands to train a model are as follows (you can e.g. run this in a Python notebook).

```
import ecog2txt.trainers as e2t_trainers
import ecog2txt.data_generators

# CREATE A NEW MODEL
trainer = e2t_trainers.MultiSubjectTrainer(
    experiment_manifest_name=my_experiment_manifest.yaml,
    subject_ids=[400, 401],
    SN_kwargs={
        'FF_dropout': 0.4,          # overwriting whatever is in the manifest
        'TEMPORALLY_CONVOLVE': True # overwriting whatever is in the manifest
    },
    DG_kwargs={
        'REFERENCE_BIPOLAR': True,  # overwriting whatever is in the manifest
    },
    ES_kwargs = {
        'data_mapping': {           # overwriting whatever is in the manifest
            'encoder_inputs': 'ecog_sequence',
            'decoder_targets': 'text_sequence',
        },
    },
)

# MAKE SURE ALL THE TFRECORDS ARE WRITTEN
for subject in trainer.ecog_subjects:
    subject.write_tf_records_maybe()
trainer.subject_to_table()

# TRAIN THE TWO SUBJECTS IN PARALLEL
assessments = trainer.parallel_transfer_learn()
```
