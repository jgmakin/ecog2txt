# standard libraries
import pdb
import os
import re
import yaml
from functools import reduce
from collections import defaultdict


# third-party packages
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

# local
from ..utils_jgm.toolbox import heatmap_confusions, MutableNamedTuple
from ..machine_learning.neural_networks import sequence_networks
from ..machine_learning.neural_networks import tf_helpers as tfh
from ..machine_learning.neural_networks import basic_components as nn

from .subjects import ECoGSubject
from . import plotters
from . import text_dir
from . import EOS_token, pad_token, OOV_token, TOKEN_TYPES, DATA_PARTITIONS


'''
:Author: J.G. Makin (except where otherwise noted)
'''


class MultiSubjectTrainer:
    def __init__(
        self,
        experiment_manifest_name,
        subject_ids,
        checkpoint_dir='',
        restore_epoch=None,
        unique_targets_list=None,
        unique_encoder_targets_list=None,
        SN_kwargs=(),
        DG_kwargs=(),
        RP_kwargs=(),
        ES_kwargs=(),
        VERBOSE=True
    ):

        # load the experiment_manifest
        with open(os.path.join(text_dir, experiment_manifest_name)) as file:
            self.experiment_manifest = yaml.full_load(file)

        # checks
        token_type = self.experiment_manifest[subject_ids[-1]]['token_type']
        assert token_type in TOKEN_TYPES, 'Unrecognized token_type!! -- jgm'

        # attribute
        self._token_type = token_type   # NB: changes will not propagate
        self._RP_kwargs = dict(RP_kwargs)

        # create ECoG subjects
        self.ecog_subjects = [
            ECoGSubject(
                self.experiment_manifest[subject_id],
                subject_id,
                pretrain_all_blocks=(subject_id != subject_ids[-1]),
                **dict(ES_kwargs),
                _DG_kwargs=dict(DG_kwargs)
                #####
                # target_specs=target_specs
                #####
            ) for subject_id in subject_ids]

        # create the SequenceNetwork according to the experiment_manifest
        self.net = sequence_networks.SequenceNetwork(
            self.experiment_manifest[subject_ids[-1]],
            EOS_token=EOS_token,
            pad_token=pad_token,
            OOV_token=OOV_token,
            training_GPUs=[0],
            TARGETS_ARE_SEQUENCES='sequence' in token_type,
            VERBOSE=VERBOSE,
            **dict(SN_kwargs)
        )

        # invoke some setters
        # NB: these attributes adjust self.ecog_subjects and self.net, so they
        #  must be invoked *after* those are created.  Hence no auto_attribute!
        self.VERBOSE = VERBOSE
        self.checkpoint_dir = checkpoint_dir
        self.restore_epoch = restore_epoch
        self.unique_targets_list = unique_targets_list
        self.unique_encoder_targets_list = unique_encoder_targets_list

        for subject in self.ecog_subjects:

            # adjust data_manifests for the specifics of this experiment
            for key, data_manifest in subject.data_manifests.items():
                # see if the experiment_manifest specifies a penalty_scale
                seq_type = data_manifest.sequence_type
                try:
                    data_manifest.penalty_scale = self.experiment_manifest[
                        subject.subnet_id][seq_type + '_penalty_scale']
                except KeyError:
                    pass

                # the decoder targets are the unique_targets_list
                if key == 'decoder_targets':
                    data_manifest.get_feature_list = (
                        lambda: self.unique_targets_list
                    )

                # if the encoder_targets are text...
                if (
                    key == 'encoder_targets' and
                    seq_type in ['phoneme_sequence', 'text_sequence']
                ):
                    # ...then then are in the unique_encoder_targets_list
                    data_manifest.get_feature_list = (
                        lambda: self.unique_encoder_targets_list
                    )

                else:
                    # there are other conceivable possibilities...
                    pass

    def vprint(self, *args, **kwargs):
        if self.VERBOSE:
            print(*args, **kwargs)

    @property
    def checkpoint_dir(self):

        # update the SequenceNetwork's checkpoint_path as well!
        self.net.checkpoint_path = os.path.join(
            self._checkpoint_dir, 'model.ckpt'
        )
        return self._checkpoint_dir

    @checkpoint_dir.setter
    def checkpoint_dir(self, checkpoint_dir):

        # set the shadow variable
        self._checkpoint_dir = checkpoint_dir

        # make sure the self.net.checkpoint_path gets updated as well
        self.checkpoint_dir

    @property
    def restore_epoch(self):
        if self._restore_epoch is not None:
            return self._restore_epoch
        else:
            model_name = 'model.ckpt'
            restore_epochs = [
                int(name.split('-')[1].split('.')[0])
                for name in os.listdir(self.checkpoint_dir)
                if name.split('-')[0] == model_name and
                name.split('.')[-1] == 'index'
            ]
            restore_epochs.sort()
            if restore_epochs:
                return restore_epochs[-1]
            else:
                # no models have been trained yet!
                return None

    @restore_epoch.setter
    def restore_epoch(self, restore_epoch):
        self._restore_epoch = restore_epoch

    @property
    def unique_targets_list(self):
        '''
        Priority order:
            (1) UTL passed as argument or set as attribute
            (2) UTL constructed from vocab_file passed or set
            (3) UTL loaded from 'unique_targets.pkl' in the checkpoint_dir
            (4) UTL constructed by union of: the intersection (across subjects)
                of training data and the union (across subjects) of validation
                data
        '''

        UTL_path = os.path.join(self.checkpoint_dir, 'unique_targets.pkl')

        if self._unique_targets_list:
            self.vprint('USING _UNIQUE_TARGETS_LIST')
            UTL = self._unique_targets_list
        elif self.ecog_subjects[-1].data_generator.vocab_file:
            # as long as the final subject has a path to a vocab file, use it
            self.vprint('CONSTRUCTING UNIQUE_TARGETS_LIST FROM VOCAB_FILE')
            UTL = self.ecog_subjects[-1].data_generator.get_unique_targets()
        elif os.path.isfile(UTL_path):
            self.vprint('LOADING UNIQUE_TARGETS_LIST FROM PICKLE FILE')
            with open(UTL_path, 'rb') as fp:
                unique_bytes_list = pickle.load(fp)
            UTL = [t.decode('utf-8') for t in unique_bytes_list]
        else:
            self.vprint('CONSTRUCTING UNIQUE_TARGETS_LIST VIA ', end='')
            self.vprint('TRAINING/INTERSECTION, VALIDATION/UNION')
            UTL = self._training_intersection_validation_union(
                'decoder_targets',
                ([pad_token, EOS_token, OOV_token]
                 if 'sequence' in self._token_type else
                 [OOV_token])
            )
        self.vprint('There are %i unique token types' % len(UTL))

        if 'sequence' in self._token_type:
            assert EOS_token in UTL, "Sequence data require an EOS_token"
            assert pad_token in UTL, "Sequence data require a pad_token"

        # You don't want to have to keep re-acquiring this list, so store
        #  it here.  If you do want to re-calculate it, set it to None.
        self._unique_targets_list = UTL

        return UTL

    @unique_targets_list.setter
    def unique_targets_list(self, unique_targets_list):
        self._unique_targets_list = unique_targets_list

    @property
    def unique_encoder_targets_list(self):
        if self._unique_encoder_targets_list:
            self.vprint('USING _UNIQUE_ENCODER_TARGETS_LIST')
            UETL = self._unique_encoder_targets_list
        else:
            self.vprint('CONSTRUCTING UNIQUE_ENCODER_TARGETS_LIST VIA', end='')
            self.vprint(' TRAINING/INTERSECTION, VALIDATION/UNION')
            UETL = self._training_intersection_validation_union(
                'encoder_targets', [pad_token])
        self.vprint('There are %i unique phonemes' % len(UETL))

        # You don't want to have to keep re-acquiring this list, so store
        #  it here.  If you do want to re-calculate it, set it to None.
        self._unique_encoder_targets_list = UETL

        return UETL

    @unique_encoder_targets_list.setter
    def unique_encoder_targets_list(self, unique_encoder_targets_list):
        self._unique_encoder_targets_list = unique_encoder_targets_list

    @property
    def results_plotter(self):
        subject = self.ecog_subjects[-1]
        results_plotter = plotters.ResultsPlotter(
            self.experiment_manifest[subject.subnet_id], subject,
            VERBOSE=self.VERBOSE, **self._RP_kwargs
        )

        # set up methods
        results_plotter.get_saliencies = self.get_saliencies
        results_plotter.get_encoder_embedding = self.get_encoder_embedding
        results_plotter.get_internal_activations = self.get_internal_activations
        results_plotter.get_sequence_counters = \
            lambda threshold: subject.get_unique_target_lengths(
                self.unique_targets_list, threshold
            )

        return results_plotter

    def parallel_transfer_learn(self, RESUME=False, fit_kwargs=()):
        '''
        Parallel transfer learning
        '''

        if RESUME:
            fit_kwargs = {
                '_restore_epoch': self.restore_epoch,
                **dict(fit_kwargs),
                'train_vars_scope': 'seq2seq',
                'reuse_vars_scope': 'seq2seq',
            }
            self.ecog_subjects = [self.ecog_subjects[-1]]

        # fit and save the results
        assessments = self.net.fit(
            self.unique_targets_list, self.ecog_subjects, **dict(fit_kwargs))
        self._save_results(assessments)

        # to facilitate restoring/assessing, update hard-coded restore_epochs
        if self._restore_epoch is not None:
            self.restore_epoch = (self.restore_epoch + self.net.Nepochs
                                  if RESUME else self.net.Nepochs)

        return assessments

    def sequential_transfer_learn(
        self, pretraining_epochs=60, training_epochs=200, posttraining_epochs=340
    ):
        '''
        Sequential transfer learning.
        '''

        # set which layers are frozen, reused, reinitialized
        proprietary_scopes = 'seq2seq/subnet'
        reusable_scopes = 'seq2seq/(?!subnet)'  # negative lookahead

        # train on each subject sequentially
        fit_kwargs = {}
        for subject in self.ecog_subjects:

            # pre-training
            if subject == self.ecog_subjects[0]:
                # first subject; do nothing but set up for next training phase
                latest_epoch = 0
                fit_kwargs['reuse_vars_scope'] = None
            else:
                # first acquire this subject's encoder embedding
                self.net.Nepochs = pretraining_epochs
                fit_kwargs['train_vars_scope'] = proprietary_scopes
                fit_kwargs['reuse_vars_scope'] = reusable_scopes
                fit_kwargs['_restore_epoch'] = latest_epoch
                self.net.fit(self.unique_targets_list, [subject], **fit_kwargs)

                # then set up for next next training phase
                latest_epoch += self.net.Nepochs
                fit_kwargs['_restore_epoch'] = latest_epoch
                fit_kwargs['reuse_vars_scope'] = 'seq2seq'

            # full training
            if subject == self.ecog_subjects[-1]:
                training_epochs += posttraining_epochs
            self.net.Nepochs = training_epochs
            fit_kwargs['train_vars_scope'] = 'seq2seq'
            assessments = self.net.fit(
                self.unique_targets_list, [subject], **fit_kwargs)
            latest_epoch += self.net.Nepochs
            self._save_results(assessments)

        # to facilitate restoring and assessing, store this
        self.restore_epoch = latest_epoch

        return assessments

    def assess_saved_model(self):

        self.update_net_from_saved_model()
        assessment_dict = self.net.restore_and_assess(
            self.unique_targets_list, self.ecog_subjects, self.restore_epoch)
        return assessment_dict

    def update_net_from_saved_model(self):
        # pull the model sizes from the saved file
        self.net.layer_sizes, data_sizes, strides, EMA = self.recover_model_sizes()
        self.net.TEMPORALLY_CONVOLVE = len(strides)
        self.net.EMA_decay = 0.99*EMA

        # these vary by subject
        for subject in self.ecog_subjects:
            s_id = subject.subnet_id
            manifests = subject.data_manifests

            #######
            # This can fail for a non-standard data_mapping
            for key, data_size in data_sizes[s_id].items():
                manifests[key].num_features = data_size
            #######

            # data sizes that hold for all subjects use the key None
            if None in data_sizes.keys():
                for key, data_size in data_sizes[None].items():
                    manifests[key].num_features = data_size

            # convolutional?
            if strides[s_id]:
                subject.decimation_factor = np.prod(strides[s_id])
            # otherwise go with the default value

    def _training_intersection_validation_union(
        self, data_key, special_tokens=[]
    ):
        '''
        Typically used when neither a UTL nor a vocab_file has been provided
        '''

        # to get the unique_targets_list...
        targets_list = list(reduce(
            # ...reduce via the union across the DATA_PARTITIONS...
            lambda A, B: A | B, [
                reduce(
                    # ...of the reductions across the intersection or union...
                    (lambda A, B: A & B) if data_partition == 'training' else (
                        lambda A, B: A | B),
                    # ...of the unique_targets_list of this data_partition
                    [
                        set(s.write_tf_records_maybe([data_partition], data_key))
                        for s in self.ecog_subjects
                    ]
                ) for data_partition in DATA_PARTITIONS
            ]
        ))
        self.vprint('All tf_records have been written...')

        # insert at the beginning, and in order, any special_tokens
        targets_list = [t for t in targets_list if t not in special_tokens]
        for token in reversed(special_tokens):
            targets_list.insert(0, token)

        return targets_list

    def recover_model_sizes(self):
        #####
        # TO DO:
        # (1) maybe this should be put into SequenceNets, since it's hard-coded
        #   for that particular network....
        #####

        # extract the dictionary mapping long var names to shapes
        reader = pywrap_tensorflow.NewCheckpointReader(os.path.join(
            self.checkpoint_dir, 'model.ckpt') + '-%i' % 10)
        var_to_shape = reader.get_variable_to_shape_map()

        # Accumulate a useful structure of network sizes.  You have to assemble
        #  the intermediate data structure before unpacking into
        #  the ones that will be returned because to put the layer sizes
        #  in order you need to collect all of them first.
        net_info = defaultdict(lambda: defaultdict(dict))
        EMA = False
        for var_name, var_shape in var_to_shape.items():
            name_scopes = var_name.split('/')
            outer_scope = name_scopes.pop(0)

            # note if an exponential moving average was used
            if name_scopes[-1] == 'ExponentialMovingAverage':
                EMA = True

            if outer_scope == 'seq2seq':
                subsubnet = name_scopes.pop(0)

                # if this is a subnetwork, find out which one
                if re.match('subnet_\d*', subsubnet):
                    subnet_id = subsubnet.split('_')[1]
                    subsubnet = name_scopes.pop(0)
                else:
                    subnet_id = None

                # check if it's an RNN
                for scope in name_scopes:
                    match_obj = re.match('cell_\d*', scope)
                    if match_obj:
                        layer_number = int(match_obj[0].split('_')[-1])
                        break
                else:
                    # it's not an RNN
                    if name_scopes[0] == 'weights':
                        # there are three numbers appended to each name
                        subsubnet, _, _, layer_number = subsubnet.rsplit('_', 3)
                        layer_number = int(layer_number)
                    else:
                        continue

                # store
                net_info[subnet_id][subsubnet][layer_number] = var_shape

        # Now unpack into data structures useful for a SequenceNetwork
        layer_sizes = {}
        data_sizes = defaultdict(dict)
        encoder_strides = defaultdict(list)

        for subnet_id, subnet_info in net_info.items():
            for subsubnet, subsubnet_info in subnet_info.items():

                # We *assume* (given the implementation of SequenceNets) that
                #  the layer_sizes do not vary across subjects/subnet_ids! so
                #  only the last subnet_id will count.
                layer_sizes[subsubnet] = []

                for layer_number in sorted(subsubnet_info.keys()):

                    # the final projection layer is special
                    if '_projection' in subsubnet and layer_number == max(
                        subsubnet_info.keys()
                    ):
                        # The only relevant info is the output size.  NB that
                        # the weight matrix of this layer is *transposed*
                        data_sizes[subnet_id][subsubnet.replace('_projection', '_target')] = \
                            subsubnet_info[layer_number][0]
                    else:
                        weight_shape = subsubnet_info[layer_number]

                        # the base layer size (may need to be adjusted)
                        layer_size = weight_shape[-1]

                        # the LSTM variables pack together 4 weight matrices
                        if '_rnn' in subsubnet:
                            layer_size //= 4

                        # add this layer size to the current list
                        layer_sizes[subsubnet].append(layer_size)

                    # the encoder_embedding is special
                    if subsubnet == 'encoder_embedding':
                        if len(weight_shape) == 4:
                            encoder_strides[subnet_id].append(weight_shape[1])

                        # 1st encoder_embedding layer has info about input size
                        if layer_number == min(subsubnet_info.keys()):
                            data_sizes[subnet_id]['encoder_inputs'] = weight_shape[-2]

        return layer_sizes, data_sizes, encoder_strides, EMA

    def _save_results(self, assessments):
        '''
        Write out to a text file
        '''

        # the save-file path/name
        experiment_manifest = self.experiment_manifest[
            self.ecog_subjects[-1].subnet_id]
        save_file_dir = experiment_manifest['saved_results_dir']
        project = experiment_manifest['project']
        save_file_path = os.path.join(
            save_file_dir,
            '_'.join(
                [
                    'accuracies',
                    project + '-'.join(str(s.subnet_id) for s in self.ecog_subjects),
                    str(self.net.FF_dropout),
                    str(self.net.RNN_dropout),
                ] + [
                    '-'.join(str(N) for N in self.net.layer_sizes[key])
                    # fix the *order* of the keys
                    for key in [
                        'encoder_embedding', 'preencoder_rnn', 'encoder_rnn',
                        'encoder_projection', 'decoder_embedding',
                        'decoder_rnn', 'decoder_projection'
                    ]
                ]
            )
        )
        print('save file is ' + save_file_path)

        # variables used for for plotting
        plot_interval = self.net.assessment_epoch_interval
        max_epoch = len(assessments['training'].accuracies)*plot_interval
        accuracies_epochs = [epoch for epoch in range(0, max_epoch, plot_interval)]

        # save the accuracies to a text file
        np.savetxt(save_file_path,
                   np.stack([assessments['training'].accuracies,
                             assessments['training'].word_error_rates,
                             assessments['validation'].accuracies,
                             assessments['validation'].word_error_rates,
                             np.array(accuracies_epochs)], axis=1),
                   fmt="%.4f",
                   header=('training accs | training WERs | '
                           'validation acc | validation WERs | epochs')
                   )

        # confusion matrix looks bad in tensorboard, so rebuild here
        N = self.ecog_subjects[-1].data_manifests['decoder_targets'].num_features
        if N < 100:
            fig_dimension = N//6
            confusions = assessments['validation'].confusions
            if confusions is not None:
                fig = heatmap_confusions(
                    plt.figure(figsize=(fig_dimension, fig_dimension)),
                    confusions,
                    x_axis_labels=self.unique_targets_list,
                    y_axis_labels=self.unique_targets_list,
                )
                fig.savefig(os.path.join(
                    save_file_dir, '%s_confusions.pdf' % self._token_type),
                    bbox_inches='tight')

    def count_all_targets(self, threshold=0.4):

        # dump into two tuples (each entry in a tuple corresponds to a subject)
        target_counters, sequence_counters = zip(*[subj.count_targets(
            self.unique_targets_list, threshold) for subj in self.ecog_subjects])

        # convert tuples into dictionaries so we know which subject is which
        def tuple_to_dict(tpl):
            return {s.subnet_id: t for (s, t) in zip(self.ecog_subjects, tpl)}
        return tuple_to_dict(target_counters), tuple_to_dict(sequence_counters)

    def subject_to_table(self):
        subject_attributes = {
            'block_types',
            'block_ids',
            'decimation_factor',
        }
        trainer_attributes = {
            # 'checkpoint_dir',
            'restore_epoch',
            'unique_targets_list',
            # 'vocab_file',
        }
        params_series = [pd.Series(
            {
                # **{k: v for k, v in s.__dict__.items() if not k.startswith('_')},
                **{key: manifest.num_features
                   for key, manifest in s.data_manifests.items()},
                **{attr: getattr(s, attr) for attr in subject_attributes},
                **{attr: getattr(self, attr) for attr in trainer_attributes},
            },
            name=s.subnet_id) for s in self.ecog_subjects
        ]
        #pdb.set_trace()
        return pd.concat(params_series, axis=1).transpose()

    def print_tensor_names(self):
        ckpt = os.path.join(self.checkpoint_dir, 'model.ckpt') + '-' + repr(
            self.restore_epoch)
        print_tensors_in_checkpoint_file(
            file_name=ckpt,
            tensor_name='',
            all_tensors=False,
            all_tensor_names=False
        )

    def cluster_embedded_words(self, weights_name, cluster_embeddings_kwargs=()):
        W = self._retrieve_layer_weights(weights_name)
        return plotters.cluster_embeddings(W, **cluster_embeddings_kwargs)

    def _retrieve_layer_weights(self, weights_name):

        # assemble the full name of the weights
        reader = pywrap_tensorflow.NewCheckpointReader(
            self.net.checkpoint_path + '-%i' % self.restore_epoch)
        var_to_shape = reader.get_variable_to_shape_map()
        weights_full_name = None
        for key in sorted(var_to_shape):
            ####
            # This isn't really right: the 0 says to use the *first* layer of
            #  whatever part of the network, but you really should use the first
            #  for the embedding and the last for the projection....
            if re.match('.*{0}.*0/weights/ExponentialMovingAverage'.format(
                    weights_name), key):
                weights_full_name = key
        assert weights_full_name, "Uh-oh, no such weights found! -- jgm"

        # extract this weight
        W = self.net.get_weights_as_numpy_array(
            weights_full_name, self.restore_epoch)
        return W

    def get_saliencies(self, contrib_method, assessment_type='norms'):
        '''
        Compute average "saliency" of input electrodes by backpropagating
        error gradients into the inputs.
        '''

        # save the original penalties in a temporary variable
        old_penalties = {}
        subject = self.ecog_subjects[-1]
        for key, manifest in subject.data_manifests.items():
            if '_targets' in key:
                old_penalties[key] = manifest.penalty_scale
                manifest.penalty_scale = 0.0

        # set the penalty for the output under consideration to 1.0
        key = contrib_method.replace('saliency_map', 'targets')
        subject.data_manifests[key].penalty_scale = 1.0

        # backpropagate error derivatives into the inputs
        contributions = self.net.restore_and_get_saliencies(
            self.unique_targets_list, [subject], self.restore_epoch,
            data_partition='validation', assessment_type=assessment_type)

        # set the penalties back to their original value
        for key, manifest in subject.data_manifests.items():
            if '_targets' in key:
                manifest.penalty_scale = old_penalties[key]

        return contributions

    def get_encoder_embedding(self):
        # fixed properties
        embedding_partial_name = (
            'seq2seq/subnet_{0}/encoder_embedding_{1}_{2}_0'
            '/weights/ExponentialMovingAverage'
        )

        # first get the *name* of the weight matrix, based on its size
        layer_sizes, data_sizes, _, _ = self.recover_model_sizes()
        embedding_name = embedding_partial_name.format(
            self.subj_id,
            data_sizes[self.subj_id]['encoder_input'],
            layer_sizes['encoder_embedding'][0]
        )

        # then get that matrix
        return self.net.get_weights_as_numpy_array(
            embedding_name, self.restore_epoch)

    ######
    # You should make it easier to do what you do here.  E.g., there should be
    #  a more general way to make an appropriate AssessmentTuple.
    ######
    def get_internal_activations(self):
        # You should make these arguments--although that would require getting
        #  some other things to work....
        op_strings = [
            'convolved_inputs',
            'reversed_inputs',
            'decimated_reversed_targets',
            'final_RNN_state',
        ]

        # ...
        subnet_params = self.ecog_subjects[-1]

        class BriefAssessmentTuple(MutableNamedTuple):
            __slots__ = ['initializer'] + op_strings

        def assessment_data_fxn(num_epochs):
            GPU_op_dict, CPU_op_dict, assessments = \
                self.net._generate_oneshot_datasets(
                    self.unique_targets_list, subnet_params, 0
                )
            brief_assessments = {
                'validation': BriefAssessmentTuple(
                    initializer=assessments['validation'].initializer,
                    **{op_string: None for op_string in op_strings}
                )
            }
            return GPU_op_dict, CPU_op_dict, brief_assessments

        def assessment_net_builder(GPU_op_dict, CPU_op_dict):
            with tf.variable_scope('seq2seq', reuse=tf.compat.v1.AUTO_REUSE):
                # reverse and decimate encoder targets
                _, get_targets_lengths = nn.sequences_tools(
                    GPU_op_dict['encoder_targets'])
                reverse_targets = tf.reverse_sequence(
                    GPU_op_dict['encoder_targets'], get_targets_lengths,
                    seq_axis=1, batch_axis=0)
                decimate_reversed_targets = reverse_targets[
                    :, 0::subnet_params.decimation_factor, :]

                self.net._prepare_encoder_targets(
                    GPU_op_dict, 0, subnet_params.decimation_factor)

                with tf.compat.v1.variable_scope(
                    'subnet_{}'.format(subnet_params.subnet_id,),
                    reuse=tf.compat.v1.AUTO_REUSE
                ):
                    # reverse inputs
                    _, get_lengths = nn.sequences_tools(tfh.hide_shape(
                        GPU_op_dict['encoder_inputs']))
                    reverse_inputs = tf.reverse_sequence(
                        GPU_op_dict['encoder_inputs'], get_lengths,
                        seq_axis=1, batch_axis=0)

                    # convolve inputs
                    convolve_reversed_inputs, _ = self.net._convolve_sequences(
                        reverse_inputs, subnet_params.decimation_factor,
                        subnet_params.data_manifests['encoder_inputs'].num_features,
                        self.net.layer_sizes['encoder_embedding'], 0.0,
                        'encoder_embedding', tower_name=''
                    )

                # get the encoder state
                get_final_state, _, _, _, _ = self.net._encode_sequences(
                    GPU_op_dict['encoder_inputs'], subnet_params, 0.0, 0.0,
                    set_initial_ind=0)

            # give names to these so you can recover them later
            decimate_reversed_targets = tf.identity(
                decimate_reversed_targets, 'assess_decimated_reversed_targets')
            convolve_reversed_inputs = tf.identity(
                convolve_reversed_inputs, 'assess_convolved_inputs')
            reverse_inputs = tf.identity(
                reverse_inputs, 'assess_reversed_inputs')
            get_final_state = tf.identity(
                get_final_state, 'assess_final_RNN_state')

            # one day you will be able to get rid of these...
            return None, None

        def assessor(
            sess, assessment_struct, epoch, assessment_step, data_partition
        ):
            sess.run(assessment_struct.initializer)
            assessments = sess.run([
                sess.graph.get_operation_by_name('assess_' + op_string).outputs[0]
                for op_string in op_strings]
            )
            for op_string, assessment in zip(op_strings, assessments):
                setattr(assessment_struct, op_string, assessment)

            return assessment_struct

        # use the general graph build to assemble these pieces
        graph_builder = tfh.GraphBuilder(
            None, assessment_data_fxn, None, assessment_net_builder, None,
            assessor, self.net.checkpoint_path, self.restore_epoch,
            self.restore_epoch-1, EMA_decay=self.net.EMA_decay,
            assessment_GPU=self.net.assessment_GPU,
        )

        return graph_builder.assess()


def construct_online_predictor(
    restore_dir, unique_targets_list=None, TARGETS_ARE_SEQUENCES=False
):

    # open a session with the saved_model loaded into it
    sess = tfh.get_session_with_saved_model(restore_dir)

    # create a function which uses this session to decode
    def predict(inputs):
        decoded_probs, sequenced_decoder_outputs = sess.run(
            ['decoder_probs:0', 'sequenced_decoder_outputs:0'],
            feed_dict={'encoder_inputs:0': inputs}
        )
        if unique_targets_list:
            unique_tokens = (
                unique_targets_list if TARGETS_ARE_SEQUENCES else
                nn.targets_to_tokens(unique_targets_list, pad_token))
            hypotheses = target_inds_to_sequences(
                sequenced_decoder_outputs, unique_tokens)[0]
            return hypotheses
        else:
            return decoded_probs

    return predict


def target_inds_to_sequences(hypotheses, targets_list, iExample=0):
    ######
    # This is redundant with the one in SequenceNets.  Think about
    #  the best place to put a single version....
    ######
    predicted_tokens = [
        ''.join([targets_list[ind] for ind in hypothesis]).replace(
            '_', ' ').replace(pad_token, '').replace(
            EOS_token, '').rstrip()
        for hypothesis in hypotheses[iExample]
    ]
    return predicted_tokens


# not currently in use
def default_batch_size(machine_name, target_type):
    sample_dict = {}
    sample_dict['domestica', 'Phoneme'] = 1024  # fix
    sample_dict['domestica', 'Word'] = 1024
    sample_dict['domestica', 'Trial'] = 20  # fix

    sample_dict['CUPCAKE', 'Phoneme'] = 512  # fix
    sample_dict['CUPCAKE', 'Word'] = 320
    sample_dict['CUPCAKE', 'Trial'] = 20  # fix

    return sample_dict[machine_name, target_type]
