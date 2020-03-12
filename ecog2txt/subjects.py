# standard libraries
import pdb
import os
import json
import copy

# third-party packages
import numpy as np
import tensorflow as tf

# local
from utils_jgm.toolbox import wer_vector, auto_attribute, str2int_hook
from machine_learning.neural_networks import tf_helpers as tfh
from ecog2txt import EOS_token, pad_token, OOV_token, DATA_PARTITIONS


'''
:Author: J.G. Makin (except where otherwise noted)
Split from trainers.py:      01/21/20
'''


class ECoGSubject:
    @auto_attribute(CHECK_MANIFEST=True)
    def __init__(
        self,
        manifest,
        subj_id,
        pretrain_all_blocks=False,
        input_mask=None,
        target_specs=(),
        block_ids=(),
        decimation_factor=None,
        #####
        # in the manifest
        block_types=None,
        data_mapping=None,
        sampling_rate_decimated=None,
        #####
        # private; do no assign to self
        _DG_kwargs=(),
    ):
        '''
        An ECoGSubject is mostly a collection of attributes, but also includes
        an ECoGDataGenerator. Most (but not all) of the attributes are intended
        for accesss by a SequenceNetwork; in particular, a list of ECoGSubjects
        (or any other object with the parameters listed below) can be passed as
        an element of a list `params` to the SequenceNetwork's fit method. Such
        a list is created by (and attributed to) a MultiSubjectTrainer.

        Attributes intended for a SequenceNetwork:
            subnet_id
            block_ids
            decimation_factor
            tf_record_partial_path
            input_mask***
            target_specs***

        Other attributes include:
            data_generators             (for generating data!)
            block_types                 (for constructing block_ids)
            pretrain_all_blocks         (for constructing block_ids)
        '''

        # get the block_breakdowns
        json_dir = manifest['json_dir']
        with open(os.path.join(json_dir, 'block_breakdowns.json')) as f:
            block_breakdowns = json.load(f, object_hook=str2int_hook)[subj_id]
        self._block_dict = block_breakdowns

        # these attributes will *not* be accessed by a SequenceNet
        DataGenerator = manifest['DataGenerator']
        self.data_generator = DataGenerator(manifest, subj_id, **dict(_DG_kwargs))

        # these attribute *will* be accessed by a SequenceNet
        self.target_specs = dict(target_specs)
        self.data_manifests = {
            key: SequenceDataManifest(sequence_type)
            for key, sequence_type in self.data_mapping.items()
        }

    # ATTRIBUTES THAT WILL *NOT* BE ACCESSED BY A SequenceNet
    @property
    def input_mask(self):
        return self._input_mask

    @input_mask.setter
    def input_mask(self, input_mask):

        # assign the shadow variable
        self._input_mask = input_mask

        # make this input mask consistent with this subject
        if self._input_mask is not None:
            self._input_mask.good_channels = self.data_generator.good_channels

    # ATTRIBUTES THAT *WILL* BE ACCESSED BY A SequenceNet
    @property
    def subnet_id(self):
        return self.subj_id

    @property
    def block_ids(self):

        if self._block_ids:
            return self._block_ids
        else:
            block_ids = {
                data_partition: {
                    blk for blk in self._block_dict if
                    self._block_dict[blk]['default_dataset'] == data_partition and
                    self._block_dict[blk]['type'] in self.block_types[data_partition]
                } for data_partition in DATA_PARTITIONS
            }
            if self.pretrain_all_blocks:
                block_ids['training'] = {
                    blk for blk_list in block_ids.values() for blk in blk_list
                }

            # if we will be specifying targets...
            if self.target_specs:
                # ... then we assign all blocks to all partitions
                blocks = {blk for blks in block_ids.values() for blk in blks}
                block_ids = {partition: blocks for partition in DATA_PARTITIONS}

            return block_ids

    @block_ids.setter
    def block_ids(self, block_ids):
        self._block_ids = block_ids

    @property
    def tf_record_partial_path(self):
        return self.data_generator.tf_record_partial_path

    @property
    def decimation_factor(self):
        if self._decimation_factor:
            factor = self._decimation_factor
        else:
            factor = int(np.round(
                self.data_generator.sampling_rate/self.sampling_rate_decimated))

        return factor

    @decimation_factor.setter
    def decimation_factor(self, decimation_factor):
        self._decimation_factor = decimation_factor

    @property
    def data_manifests(self):
        for data_manifest in self._data_manifests.values():
            if data_manifest.sequence_type == 'ecog_sequence':
                data_manifest.num_features = self.data_generator.num_channels
            elif data_manifest.sequence_type == 'text_sequence':
                # unfortunately, the ingredients to set this aren't here
                pass
            elif data_manifest.sequence_type == 'audio_sequence':
                data_manifest.num_features = \
                    self.data_generator.num_MFCC_features
            elif data_manifest.sequence_type == 'phoneme_sequence':
                # unfortunately, the ingredients to set this aren't here
                pass
            else:
                raise ValueError('unexpected data_manifest sequence_type')

        return self._data_manifests

    def unique_targets_list(self):
        print('inside UTL')
        return []

    @data_manifests.setter
    def data_manifests(self, data_manifests):
        self._data_manifests = data_manifests

    def write_tf_records_maybe(
        self, data_partitions=DATA_PARTITIONS, data_key='decoder_targets'
    ):
        # NB: if there is a vocab_file, then it doesn't matter what the
        #  manifest data_key is set to: the contents of the vocab_file will be
        #  returned as the UTL no matter what.
        for data_partition in data_partitions:
            UTL = self.data_generator.write_to_Protobuf_maybe(
                self.block_ids[data_partition],
                self.data_manifests[data_key].sequence_type)

        # only return the UTL from the *last* data_partition
        return UTL

    def count_targets(self, unique_targets_list, threshold=0.4):

        # initialize
        target_counters = {}
        sequence_counters = {}
        unique_sequences = ()

        # do *not* transform the saved strings into indices!
        target_manifest = copy.copy(self.data_manifests['decoder_targets'])
        target_manifest.transform = None

        # for each data_partition...
        for data_partition, blocks in self.block_ids.items():
            # build two counters and apply them to *all* examples
            target_counter = TargetCounter(unique_targets_list)
            sequence_counter = SequenceCounter(unique_sequences, threshold)
            apply_to_all_tf_examples(
                [target_counter, sequence_counter],
                lambda example_proto: tfh.parse_protobuf_seq2seq_example(
                    example_proto, {'decoder_targets': target_manifest},
                ),
                blocks, self.tf_record_partial_path
            )

            # say
            print('finished count for %17s' % (data_partition), end='')
            print(' with %5i tokens,' % sum(target_counter.types), end='')
            print(' with %5i types,' % sum(target_counter.types > 0), end='')
            print(' with %5i sequence types,' % sum(
                sequence_counter.types > 0), end='')
            print(' and %5i skipped tokens' % target_counter.skipped_tokens,
                  end='')
            print(' in %5i examples' % target_counter.examples)

            # store
            target_counters[data_partition] = target_counter
            sequence_counters[data_partition] = sequence_counter

            # the next partition's sequence_counter should just add to the list
            unique_sequences = sequence_counter.unique_sequence_list

        # ...
        synchronize_sequence_counters(sequence_counters)

        return target_counters, sequence_counters

    def get_unique_target_lengths(self, unique_targets_list, threshold=0.4):

        # initialize
        sequence_counters = {}
        unique_sequence_list = ()
        for data_partition, blks in self.block_ids.items():

            # ....
            sequence_counter = SequenceCounter(
                unique_sequence_list, threshold, protobuf_name='full_record')
            apply_to_all_tf_examples(
                [sequence_counter],
                lambda example_proto: tfh.parse_protobuf_seq2seq_example(
                    example_proto, self.data_manifests
                ),
                blks, self.tf_record_partial_path
            )

            # store
            sequence_counters[data_partition] = sequence_counter

            # the next partition's sequence_counter should just add to the list
            unique_sequence_list = sequence_counter.unique_sequence_list

        # ...
        synchronize_sequence_counters(sequence_counters)

        return sequence_counters


class SequenceDataManifest:
    '''
    A simple class to hold the information for unpacking the sequence data from
    tf records--plus a few other useful values. This function can automatically
    adjust for masks that extract only a subset of the stored data.
    '''

    @auto_attribute
    def __init__(
        self,
        sequence_type,
        num_features=None,
        num_features_raw=None,
        transform=None,
        padding_value=None,
        penalty_scale=1.0,
        distribution=None,
        mask=None,
        get_feature_list=None
    ):
        pass

    @property
    def feature_value(self):
        if self.sequence_type in ['text_sequence', 'phoneme_sequence']:
            return tf.io.VarLenFeature(tf.string)
        else:
            return tf.io.VarLenFeature(tf.float32)

    @property
    def num_features(self):
        if self.mask is not None:
            return len(self.mask.inds)
        elif self.get_feature_list is not None:
            feature_list = self.get_feature_list()
            # feature_lists are for the special case of categorical data that
            #  will be converted into one-hot representations
            return len(feature_list)
        else:
            return self._num_features

    @num_features.setter
    def num_features(self, num_features):
        self._num_features = num_features

    @property
    def num_features_raw(self):
        if self._num_features_raw is not None:
            return self._num_features_raw
        elif self.mask is not None:
            # the shadow property wasn't reduced by the presence of a mask
            return self._num_features
        elif self.get_feature_list is not None:
            # feature_lists are for the special case of categorical data that
            #  will be converted into one-hot representations
            return 1
        else:
            return self.num_features

    @num_features_raw.setter
    def num_features_raw(self, num_features_raw):
        self._num_features_raw = num_features_raw

    @property
    def transform(self):
        if self._transform is not None:
            return self._transform
        elif self.mask is not None:
            return lambda seq: tfh.fancy_indexing(seq, self.input_mask.inds, 1)
        elif self.get_feature_list is not None:
            feature_list = self.get_feature_list()
            # set the ids to some defaults if they're not in the UTL
            ########
            OOV_id = (feature_list.index(OOV_token)
                      if OOV_token in feature_list else 2)
            # Just making up "2" here can give some really weird errors...
            ########
            if EOS_token in feature_list:
                # we've got sequence data
                return lambda seq: tfh.string_seq_to_index_seq(
                    seq, feature_list, [feature_list.index(EOS_token)], OOV_id,
                )
            else:
                # non-sequence data
                return lambda seq: tfh.string_seq_to_index_seq(
                    seq, feature_list, [], OOV_id
                )
        else:
            return lambda seq: seq

    @transform.setter
    def transform(self, transform):
        self._transform = transform

    @property
    def distribution(self):
        if self._distribution is not None:
            return self._distribution
        else:
            if self.sequence_type in ['text_sequence', 'phoneme_sequence']:
                return 'categorical'
            elif self.sequence_type == 'ecog_sequence':
                # but you probably don't care about the ECoG distribution...
                return 'Rayleigh'
            elif self.sequence_type == 'audio_sequence':
                return 'Gaussian'
            else:
                return 'Gaussian'

    @property
    def padding_value(self):
        if self._padding_value is None:
            if self.get_feature_list is None:
                return 0.0
            else:
                feature_list = self.get_feature_list()
                ########
                return (feature_list.index(pad_token)
                        if pad_token in feature_list else 0)
                # As above with "2," this 0 is highly dubious and will probably
                #  create difficult-to-find bugs
                ########
        else:
            return self._padding_value

    @padding_value.setter
    def padding_value(self, padding_value):
        self._padding_value = padding_value

    @distribution.setter
    def distribution(self, distribution):
        self._distribution = distribution


################
# This is probably semi-broken and in any case should be brought up to date
#  with the rest of the package.
################
class SubgridParams:
    @auto_attribute
    def __init__(
        self,
        grid_size=[16, 16],
        subgrid_size=[8, 16],
        start=[0, 0],
        SUBSAMPLE=False,
        OCCLUDE=False,
        subj_id=None,
        good_channels=None,
    ):

        # set default values
        if grid_size is None:
            self.grid_size = [16, 16]
        if subgrid_size is None:
            self.subgrid_size = [8, 16]
        if start is None:
            self.start = [0, 0]

        self.inds = None

    @property
    def _electrodes(self):
        ###########
        # This should probably use elec_layout directly....
        ###########

        # arrange electrodes in a rectilinear grid (matrix)
        full_grid_electrodes = np.reshape(
            np.arange(np.prod(self.grid_size)), self.grid_size)

        # subgrid_size is a list of either two ints or strs specifying anatomy
        ###if isinstance(subgrid_size[0], str):

        # either subsample or take a section
        if self.SUBSAMPLE:
            stop = [i+j for i, j in zip(self.start, self.grid_size)]
            step = [M//N for M, N in zip(self.grid_size, self.subgrid_size)]
        else:
            stop = [i+j for i, j in zip(self.start, self.subgrid_size)]
            step = [1, 1]

            # if "tall," the matrix must be transposed before flattening
            if self.subgrid_size[0] > self.subgrid_size[1]:
                full_grid_electrodes = full_grid_electrodes.T
                self.start.reverse()
                stop.reverse()

        return np.reshape(full_grid_electrodes[
            self.start[0]:stop[0]:step[0], self.start[1]:stop[1]:step[1]], -1)

    @property
    def inds(self):
        if self._inds is not None:
            return self._inds

        if self.good_channels is not None:
            if self.OCCLUDE:
                # only *exclude* the subgrid
                return [i for i, e in enumerate(self.good_channels)
                        if e not in self._electrodes]
            else:
                # only *include* the subgrid
                return [i for i, e in enumerate(self.good_channels)
                        if e in self._electrodes]
        else:
            return None

    @inds.setter
    def inds(self, inds):
        self._inds = inds


class TargetCounter:
    @auto_attribute
    def __init__(
        self,
        unique_targets_list,
    ):
        # the dictionary that will be updated
        self.types = np.zeros(len(unique_targets_list), dtype=int)
        self.skipped_tokens = 0
        self.examples = 0

    def update(self, byte_sequence):

        # just clean it up a bit
        sequence = [b.decode('utf-8') for b in byte_sequence]

        # all examples are counted
        self.examples += 1

        # for all entries (probably words) in this list
        for entry in sequence:
            try:
                self.types[self.unique_targets_list.index(entry)] += 1
            except ValueError:
                self.skipped_tokens += 1


class SequenceCounter:
    def __init__(
        self,
        unique_sequence_list=(),
        threshold=0.4,
        protobuf_name='decoder_targets_only'
    ):

        # attribute
        self.threshold = threshold
        self.unique_sequence_list = list(unique_sequence_list)
        self.types = np.array(
            [0 for _ in range(len(unique_sequence_list))], dtype=int)
        self.examples = 0
        self.protobuf_name = protobuf_name
        self.lengths = [[] for _ in range(len(unique_sequence_list))]

    def update(self, data_example):

        # extract the sequence as a list (of strings or indices)
        sequence = data_example['decoder_targets'][:, 0].tolist()
        if type(sequence[0]) is bytes:
            sequence = [b.decode('utf-8') for b in sequence]
        if type(sequence[0]) is str:
            sequence += [EOS_token]

        # all examples are counted
        self.examples += 1

        # if at least one sequence has been added to the list...
        if self.unique_sequence_list:
            # ...then get their word error rate from the current sequence
            WERs = wer_vector(self.unique_sequence_list,
                              [sequence]*len(self.unique_sequence_list))

            # if this sequence is close enough to an observed sequence...
            if np.min(WERs) < self.threshold:
                # ...assign it to that sequence
                self.types[np.argmin(WERs)] += 1
                if self.protobuf_name != 'decoder_targets_only':
                    self.lengths[np.argmin(WERs)].append(
                        data_example['encoder_inputs'].shape[0])
                return

        # ...otherwise append this sequence and count it
        self.unique_sequence_list.append(sequence)
        self.types = np.append(self.types, [1])
        if self.protobuf_name != 'decoder_targets_only':
            self.lengths.append([data_example['encoder_inputs'].shape[0]])

    @property
    def lengths_means(self):
        return [np.mean(lengths) for lengths in self.lengths]

    @property
    def lengths_std_errs(self):
        return [(np.var(lengths)/len(lengths))**(1/2)
                for lengths in self.lengths]


def synchronize_sequence_counters(sequence_counters):
    '''
    Enforce consistency among all sequence counters in a dictionary
    '''

    # find the partition with the longest unique_sequence_list
    max_length = -1
    for partition in DATA_PARTITIONS:
        num_sequences = len(sequence_counters[partition].unique_sequence_list)
        if num_sequences > max_length:
            max_length = num_sequences
            unique_sequences = sequence_counters[partition].unique_sequence_list

    for data_partition in DATA_PARTITIONS:
        # overwrite with the final unique_sequence_list
        sequence_counters[data_partition].unique_sequence_list = unique_sequences

        # pad out to length of final unique_sequence_list with 0s
        type_counts = sequence_counters[data_partition].types
        Npad = len(unique_sequences) - type_counts.shape[0]
        sequence_counters[data_partition].types = np.pad(
            type_counts, [0, Npad], mode='constant')

        # pad out to length of final unique_sequence_list with empty lists
        sequence_counters[data_partition].lengths.extend([[]]*Npad)


def apply_to_all_tf_examples(examplers, map_fxn, blks, tf_record_partial_path):

    tf.compat.v1.reset_default_graph()
    data_graph = tf.Graph()
    with data_graph.as_default():
        dataset = tf.data.TFRecordDataset([
            tf_record_partial_path.format(blk) for blk in blks])
        dataset = dataset.map(map_fxn, num_parallel_calls=32)
        get_next_example = dataset.make_one_shot_iterator().get_next()
        sess = tf.compat.v1.Session()
    while True:
        try:
            next_example = sess.run(get_next_example)
            for exampler in examplers:
                exampler.update(next_example)
        except tf.errors.OutOfRangeError:
            break
