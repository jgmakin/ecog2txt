# standard libraries
import pdb
import os
import json

# third-party packages
import numpy as np
from scipy.fftpack import dct
import tensorflow as tf
from tensor2tensor.data_generators import text_encoder
try:
    from python_speech_features import delta, fbank, lifter
except ModuleNotFoundError:
    pass

# local
from ..machine_learning.neural_networks import tf_helpers as tfh
from ..utils_jgm.toolbox import auto_attribute
from . import text_dir


''''
The ECoGDataGenerator class and related functions for assembling ECoG data,
and then writing them out with a generator, to a numpy tensor, or
to a tfrecord (tensorflow protobuf).

:Author: J.G. Makin (except where otherwise noted)

Created: July 2017
Revised: 02/18/20
'''

max_seconds_dict = {
    'phoneme': 0.2,
    'word': 1.0,
    'word_sequence': 6.25,
    'word_piece_sequence': 6.25,
    'phoneme_sequence': 6.25,
    'trial': 6.25
}


class ECoGDataGenerator:

    @auto_attribute(CHECK_MANIFEST=True)
    def __init__(
        self,
        manifest,
        subj_id,
        #####
        # kwargs that will default to the manifest
        grid_step=None,
        num_cepstral_coeffs=None,
        mfcc_winlen=None,
        USE_LOG_MELS=None,
        USE_MFCC_DELTAS=None,
        USE_FIELD_POTENTIALS=None,
        REFERENCE_BIPOLAR=None,
        num_mel_features=None,
        sampling_rate=None,
        token_type=None,
        vocab_file=None,
        channels_path=None,
        tf_record_partial_path=None,
        grid_size=None,
        badchannels_path=None,
        max_seconds=None,
        max_samples=None,
        good_electrodes=None,
        ######
        # private; don't assign these to self:
        # ...
    ):
        '''
        A class shell for generating ECoG data and corresponding labels in a
        format suitable for a neural network to consume.  There are two main
        methods to be used externally:
            get
            write_to_Protobuf_maybe

        To use this class, one should subclass it and provide at least these
        three methods:
            _get_wav_data
            _query
            _ecog_generator
        '''

        # set this directly to None
        self._bipolar_to_elec_map = None

    @property
    def target_type(self):
        if 'sequence' in self.token_type:
            return 'Trial'
        else:
            return self.token_type.capitalize()

    @property
    def elec_layout(self):
        layout = np.arange(np.prod(
            self.grid_size)-1, -1, -1).reshape(self.grid_size).T

        # now correct for subsampling the grid
        return layout[::self.grid_step, ::self.grid_step]

    @property
    def vocab_file_path(self):
        # The vocab file *must* live in the text_dir
        if self.vocab_file is not None:
            return os.path.join(text_dir, self.vocab_file)
        else:
            return None

    @property
    def badchannels_path(self):
        if self._badchannels_path is not None:
            return self._badchannels_path
        else:
            return os.path.join(
                os.path.join(os.path.dirname(__file__)), 'badchannels')

    @badchannels_path.setter
    def badchannels_path(self, badchannels_path):
        self._badchannels_path = badchannels_path

    @property
    def tf_record_partial_path(self):
        # something of a hack: insert a subdir before the file name
        if self.REFERENCE_BIPOLAR and self.grid_step > 1:
            subdir = 'lowdensity_bipolar'
            return os.path.join(
                os.path.dirname(self._tf_record_partial_path),
                subdir,
                os.path.basename(self._tf_record_partial_path)
            )
        else:
            return self._tf_record_partial_path

    @tf_record_partial_path.setter
    def tf_record_partial_path(self, tf_record_partial_path):
        self._tf_record_partial_path = tf_record_partial_path

    @property
    def max_seconds(self):
        # _max_seconds has precedence over the max_seconds_dict
        if self._max_seconds is not None:
            return self._max_seconds
        else:
            return max_seconds_dict.get(self.token_type, 0.2)

    @max_seconds.setter
    def max_seconds(self, max_seconds):
        self._max_seconds = max_seconds

    @property
    def max_samples(self):
        # _max_samples has precedence over max_seconds
        if self._max_samples is not None:
            return self._max_samples
        else:
            return int(np.floor(self.sampling_rate*self.max_seconds))

    @max_samples.setter
    def max_samples(self, max_samples):
        self._max_samples = max_samples

    @property
    def num_MFCC_features(self):
        if self.USE_LOG_MELS:
            return self.num_mel_features + 1
        else:
            if self.USE_MFCC_DELTAS:
                return 2*self.num_cepstral_coeffs
            else:
                return self.num_cepstral_coeffs

    @property
    def good_electrodes(self):
        '''
        NB!!! badchannels are 1-indexed, good_electrodes are zero-indexed!!

        Since this is a set, it contains no order information.  The canonical
        ordering is established with good_channels, since after all the data
        size is (... x Nchannels),  not (... x Nelectrodes).
        '''

        if self._good_electrodes is not None:
            return self._good_electrodes
        else:
            # construct by first loading the *bad*channels
            with open(self.badchannels_path, 'r') as f:
                bad_channels = f.readlines()
            bad_channels = [int(channel.strip()) for channel in bad_channels]
            return (set(range(np.prod(self.grid_size))) -
                    set(np.array(bad_channels)-1))

    @good_electrodes.setter
    def good_electrodes(self, good_electrodes):
        self._good_electrodes = good_electrodes

    @property
    def good_channels(self):
        '''
        Pseudo-channels, constructed (on the fly) from the physical electrodes.
        For now at least, we won't USE_FIELD_POTENTIALS if we want to
        REFERENCE_BIPOLAR.

        NB!!: The *order* of these channels matters--it determines the order of
        the input data, and therefore is required by the functions that plot
        electrode_contributions in plotters.py! And the order of these channels
        will be determined by the *elec_layout*.
        '''

        all_electrodes = self.elec_layout.flatten().tolist()

        if self.USE_FIELD_POTENTIALS:
            M = len(all_electrodes)
            return (
                [e for e in all_electrodes if e in self.good_electrodes] +
                [e+M for e in all_electrodes if e in self.good_electrodes]
            )
        elif self.REFERENCE_BIPOLAR:
            return [
                ch for ch, elec_pair in enumerate(self.bipolar_to_elec_map)
                if all([e in self.good_electrodes for e in elec_pair])
            ]
        else:
            return [e for e in all_electrodes if e in self.good_electrodes]

    @property
    def num_channels(self):
        return len(self.good_channels)

    def get(self, block_set):
        '''Generate and pad data'''

        # malloc the output_dict
        num_examples = self._query(block_set)
        output_dict = {
            'ecog_sequence': np.zeros(
                (num_examples, self.max_samples, self.num_channels)),
            'audio_sequence': np.zeros(
                (num_examples, self.max_samples, self.num_MFCC_features)),
            'text_sequence': [],
            'phoneme_sequence': [],
        }

        # for each block...
        i_example = 0
        print('\nLoading data for tensor construction...')
        for block in block_set:

            # ...get a lazy iterator...
            data_iterator = self._ecog_token_generator(block)

            # ...and iterate through it
            for element in data_iterator:

                # pack each entry of element into its output data_struct
                for sequence_type, data_struct in output_dict.items():
                    token = element[sequence_type]
                    if type(data_struct) is list:
                        data_struct.append(token)
                    elif type(data_struct) is np.ndarray:
                        excess = self.max_samples - token.shape[0]
                        token = np.pad(token, ((0, excess), (0, 0)), 'constant')
                        data_struct[i_example, :, :] = np.expand_dims(token, axis=0)
                    else:
                        raise ValueError('Unexpected data structure!')

                i_example += 1
        print('\n\n')

        return (
            output_dict[key] for key in [
                'ecog_sequence', 'text_sequence',
                'audio_sequence', 'phoneme_sequence'
            ]
        )

    def _write_to_Protobuf(self, block):
        '''
        Collect the relevant ECoG data and then write to disk as a (google)
         protocol buffer.
        '''
        writer = tf.io.TFRecordWriter(
            self.tf_record_partial_path.format(block))
        for example_dict in self._ecog_token_generator(block):
            feature_example = tfh.make_feature_example(example_dict)
            writer.write(feature_example.SerializeToString())

    def _ecog_token_generator(self, block):
        """A generator that yields a dictionary with:
            `ecog_sequence`: ecog data, only b/n the start and end of tokens;
            `text_sequence`: the corresponding text tokens;
            `audio_sequence`: the corresponding audio (MFCC) tokens;
            `phoneme_sequence`: ditto for phonemes--with repeats
        """

        # write to disk (by this point the grid_size and paths should be set)
        self._write_good_electrodes()

        # get the field potentials separately
        field_potentials = self._get_field_potentials(block)

        # ...
        yield from self._ecog_generator(block, field_potentials)

    def _get_MFCC_features(self, index, winstep, nfft=512):

        # first load the .wav file
        audio_sampling_rate, audio_signal = self._get_wav_data(index)

        # now convert to MFCCs
        if audio_signal is None:
            # No need to warn: that will have been done in _get_wav_data.
            # NB: This sets the MFCCs to *length-zero* sequences of vectors,
            #  each *of length num_MFCC_features*.  When called by self.get(),
            #  the sequences will anyway be padded out to self.max_samples. But
            #  when the generator is called directly, zero-length sequences
            #  will indeed by returned.
            return np.zeros((0, self.num_MFCC_features))
        elif self.num_MFCC_features == 0:
            print('WARNING: no MFCCs requested')
            # NB: You have yet to use this.  That is, in theory this allows one
            #  to request that no MFCCs be packaged with the other data; but in
            #  practice when training a SequenceNetwork w/o encoder targetting,
            #  you don't bother (you wouldn't want to have to re-create the tf
            #  records), and instead just set encoder targets penalty=0.
            Nsamples = int(audio_signal.shape[0]/audio_sampling_rate/winstep)
            return np.zeros((Nsamples, 0))
        else:
            # unpack the log-mel calculations, because you may just use them
            lowfreq = 0
            highfreq = None
            preemph = 0.97
            ceplifter = 22
            features, energy = fbank(
                audio_signal, audio_sampling_rate, self.mfcc_winlen, winstep,
                self.num_mel_features, nfft, lowfreq, highfreq, preemph,
                lambda x: np.ones((x,))
            )
            features = np.log(features)

            # use MFCCs (as opposed to log-mels)
            if not self.USE_LOG_MELS:
                features = dct(features, type=2, axis=1, norm='ortho')
                features = features[:, :self.num_cepstral_coeffs]
                features = lifter(features, ceplifter)
                features[:, 0] = np.log(energy)
            else:
                features = np.concatenate(
                    (features, np.log(energy)[:, None]), axis=1)

            # use deltas?
            mfccs = (np.concatenate((features, delta(features, N=2)), axis=1)
                     if self.USE_MFCC_DELTAS else features)

            return mfccs

    def write_to_Protobuf_maybe(self, block_set, sequence_type):

        from .subjects import SequenceDataManifest

        # set up a data manifest for loading in the text sequences
        manifest = SequenceDataManifest(sequence_type, num_features_raw=1)

        target_list = []
        for block in block_set:
            data_path = self.tf_record_partial_path.format(block)
            if not os.path.exists(data_path):
                self._write_to_Protobuf(block)

            # grab the contribution of this block to the unique_targets_list
            simple_graph = tf.Graph()
            with simple_graph.as_default():
                dataset = tf.data.TFRecordDataset(data_path)
                dataset = dataset.map(
                    lambda example_proto: tfh.parse_protobuf_seq2seq_example(
                        example_proto, {'seq': manifest}
                    )
                )
                next_example = tf.compat.v1.data.make_one_shot_iterator(
                    dataset).get_next()
                with tf.compat.v1.Session(graph=simple_graph) as sess:
                    while True:
                        try:
                            target_list.append(
                                sess.run(next_example['seq'])[:, 0].tolist()
                            )
                        except tf.errors.OutOfRangeError:
                            # print('block %i is ready' % block)
                            break
            print('.', end='')
        print()

        # bytes -> strings, and only return the unique elements
        return list(set(
            w.decode('utf-8') for word_list in target_list for w in word_list
        ))

    def get_unique_targets(self, block_set=None):
        assert (self.vocab_file_path is not None) | (block_set is not None), (
            'This ECoGDataGenerator has no vocab_file, so to generate a unique'
            ' targets list, this method must be passed a block_set'
        )
        # NB: The vocab_file has precendence over the block_set

        if self.vocab_file_path is not None:
            if self.token_type == 'word_piece_sequence':
                unique_targets_list = self.TokenEncoder()._all_subtoken_strings
            else:
                with open(self.vocab_file_path, 'r') as f:
                    unique_targets_list = [word for word in f.read().split()]
        else:
            unique_targets_list = self.write_to_Protobuf_maybe(
                block_set, 'decoder_targets'
            )

        return unique_targets_list

    def _write_good_electrodes(self):

        # get or create a channels file
        if os.path.isfile(self.channels_path):
            # file already exists; load it
            with open(self.channels_path) as f:
                num_electrodes_dict = json.load(f)
        else:
            # file doesn't exist; create dictionary from scratch
            num_electrodes_dict = {}

        # write the pseudochannels to the json file
        num_electrodes_dict[str(self.subj_id)] = sorted(list(
            self.good_electrodes))
        with open(self.channels_path, 'w') as f:
            json.dump(num_electrodes_dict, f, sort_keys=True, indent=4)

    def _sentence_tokenize(self, token_list):
        # NB that conversion to UTF-8 (bytes objects) also happens here:
        #  token_list is a list of *strings*, but the tokenized_sentence is a
        #  list of *bytes*.

        if self.vocab_file_path:
            if self.token_type == 'word_piece_sequence':
                # get the encoder and unique_targets via tensor2tensor code
                token_encoder = self.TokenEncoder()
                unique_targets = token_encoder._all_subtoken_strings

                # we can't just encode, we must also break into subwords
                indices = token_encoder.encode(' '.join(
                    [token.lower() for token in token_list]))
                tokenized_sentence = [
                    unique_targets[i].encode('utf-8') for i in indices]
            else:
                tokenized_sentence = [(token.lower() + '_').encode('utf-8')
                                      for token in token_list]
        else:
            if self.token_type == 'trial':
                # For parallelism with everything, put each sentence into a
                #  length-one list
                tokenized_sentence = [' '.join(
                    [token.lower() for token in token_list]).encode('utf-8')]
            else:
                # 'word_sequence', 'word', 'phoneme'
                tokenized_sentence = [(token.lower() + '_').encode('utf-8')
                                      for token in token_list]

        return tokenized_sentence

    def TokenEncoder(self):
        '''
        if self.token_type == 'word_piece_sequence':
            return text_encoder.SubwordTextEncoder(self.vocab_file_path)
        else:
            return text_encoder.TokenTextEncoder(
                self.vocab_file_path, replace_oov=OOV_token)
        '''
        return text_encoder.SubwordTextEncoder(self.vocab_file_path)

    #############
    # DUMMY PROPERTIES AND METHODS
    @property
    def bipolar_to_elec_map(self):
        print('WARNING!!!!  MAKING UP bipolar_to_elec_map!!!')
        elec_map = []
        layout = self.elec_layout  # for short
        for i in range(layout.shape[0]):
            for j in range(layout.shape[1]):
                if j < layout.shape[1]-1:
                    elec_map.append((layout[i, j], layout[i, j+1]))
                if i < layout.shape[0]-1:
                    elec_map.append((layout[i, j], layout[i+1, j]))
        return np.array(elec_map)

    def _get_wav_data(self, index):
        sampling_rate = None
        signal = None
        return sampling_rate, signal

    def _query(self, block_set):
        '''
        Get the number of examples for the purpose of memory pre-allocation
        '''

        num_examples = None
        return num_examples

    def _ecog_generator(self, block, field_potentials):
        '''
        Generate a dictionary of sequence data.
        '''

        for i in range(0):
            yield {
                'ecog_sequence': None,
                'text_sequence': None,
                'audio_sequence': None,
                'phoneme_sequence': None,
            }
    #############


# deprecated
def filter_to_common_targets(inputs_A, targets_A, inputs_B, targets_B):
    '''Filter out the examples that have targets not occurring in the
    other set.  For example, if the word "horse" shows up in targets_A
    but not targets_B, remove that example from targets_A and inputs_A.'''

    common_targets = set(targets_A) & set(targets_B)
    inputs_A, targets_A = filter_to_common_targets_core(
        inputs_A, targets_A, common_targets)
    inputs_B, targets_B = filter_to_common_targets_core(
        inputs_B, targets_B, common_targets)
    print('Sets (A,B) now have (%d,%d) examples and (%d,%d) unique tokens' % (
        len(targets_A), len(targets_B),
        len(set(targets_A)), len(set(targets_B))))
    return inputs_A, targets_A, inputs_B, targets_B


def filter_to_common_targets_core(inputs, targets, common_targets):

    # get the indices of training pairs whose targets are in the common set
    common_targets_indices = [ind for ind, val in enumerate(targets)
                              for this_common_target in common_targets
                              if val == this_common_target]

    # the inputs are a numpy array, the targets are a list (or list of lists)
    inputs = inputs[common_targets_indices, :, :]
    targets = [targets[ind] for ind in common_targets_indices]

    return inputs, targets

