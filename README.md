# ecog2txt
Code for decoding speech as text from neural data

This package contains Python code for the high-level aspects of decoding speech from neural data, including transfer learning across multiple subjects.  It was used to for all results in the paper "Machine translation of cortical activity to text with an encoder-decoder framework" (Makin et al., _Nature Neuroscience_, 2020).  These high-level aspects include the structuring of the training, the organization by subjects, and the construction of [tf_records](https://www.tensorflow.org/tutorials/load_data/tfrecord).  The training itself is done with adjacent [machine_learning package](https://github.com/jgmakin/machine_learning), which implements sequence-to-sequence networks in [TensorFLow](https://www.tensorflow.org).

## Getting started
In order to unify the vast set of parameters (paths, experimental block structure, neural network hyperparameters, etc.), all experiments are organized with the help of two configuration files, block_breakdowns.json, and YOUR_EXPERIMENT_manifest.yaml.  Examples of each are included in this repository.
