# standard libraries
import pdb
import os
import re
from functools import reduce, partial
from collections import defaultdict

# third-party packages
import yaml
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

# local
from utils_jgm.toolbox import heatmap_confusions, MutableNamedTuple
from machine_learning.neural_networks import tf_helpers as tfh
from ecog2txt.subjects import ECoGSubject
from ecog2txt import plotters, text_dir, TOKEN_TYPES, DATA_PARTITIONS
from ecog2txt import EOS_token, pad_token, OOV_token
if int(tf.__version__.split('.')[0]) == 2:
    # from machine_learning.neural_networks.tf_helpers_too import NeuralNetwork
    # from machine_learning.neural_networks.sequence_networks_too import Seq2Seq
    # pass
    from machine_learning.neural_networks.torch_sequence_networks import (
        Sequence2Sequence, SequenceTrainer
    )
else:
    from machine_learning.neural_networks import basic_components as nn
    from machine_learning.neural_networks.sequence_networks import SequenceNetwork


'''
:Author: J.G. Makin (except where otherwise noted)
'''


class MultiSubjectTrainer:
    def __init__(
        self,
        experiment_manifest_name,
        subject_ids,
        checkpoint_dir='.',
        restore_epoch=None,
        SN_kwargs=(),
        DG_kwargs=(),
        RP_kwargs=(),
        ES_kwargs=(),
        VERBOSE=True,
        **kwargs
    ):

        # ...
        SN_kwargs = dict(SN_kwargs)

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

        # invoke some setters
        # NB: these attributes adjust self.ecog_subjects, so they must be
        #  invoked *after* those are created (hence no auto_attribute).  But
        #  the changes to the ecog_subjects below in turn depend on the
        #  self.checkpoint_dir, so they have to be set after these lines.
        self.VERBOSE = VERBOSE
        self.checkpoint_dir = checkpoint_dir
        self.restore_epoch = restore_epoch

        # update the data_manifests for our case
        for subject in self.ecog_subjects:
            for data_key, data_manifest in subject.data_manifests.items():
                if data_key == 'decoder_targets' and 'sequence' in token_type:
                    data_manifest.APPEND_EOS = True
                try:
                    data_manifest.penalty_scale = self.experiment_manifest[
                        subject.subnet_id][data_key + '_penalty_scale']
                except KeyError:
                    pass
        self.set_feature_lists(**kwargs)

        # create the SequenceNetwork according to the experiment_manifest
        if int(tf.__version__.split('.')[0]) == 2:
            # remove SN_kwargs that aren't expected by Sequence2Sequence
            self.ST_kwargs = {
                key: SN_kwargs.pop(key) for key in {
                    'temperature', 'EMA_decay', 'beam_width',
                    'assessment_epoch_interval', 'tf_summaries_dir',
                    'N_cases',
                } if key in SN_kwargs
            }
            self.N_epochs = SN_kwargs.pop('N_epochs', None)
            self.net = Sequence2Sequence(
                self.experiment_manifest[subject_ids[-1]],
                self.ecog_subjects,
                EOS_token=EOS_token,
                pad_token=pad_token,
                # OOV_token=OOV_token,
                TARGETS_ARE_SEQUENCES='sequence' in token_type,
                VERBOSE=VERBOSE,
                **dict(SN_kwargs)
            )
        else:
            self.net = SequenceNetwork(
                self.experiment_manifest[subject_ids[-1]],
                EOS_token=EOS_token,
                pad_token=pad_token,
                OOV_token=OOV_token,
                training_GPUs=[0],
                TARGETS_ARE_SEQUENCES='sequence' in token_type,
                VERBOSE=VERBOSE,
                **dict(SN_kwargs)
            )

        # re-run to set the net's checkpoint_path
        self.checkpoint_dir = checkpoint_dir

        # initialize
        self._results_plotter = None

    def vprint(self, *args, **kwargs):
        if self.VERBOSE:
            print(*args, **kwargs)

    def set_feature_lists(self, **kwargs):
        for subject in self.ecog_subjects:

            # adjust data_manifests for the specifics of this experiment
            for data_key, data_manifest in subject.data_manifests.items():
                sequence_type = data_manifest.sequence_type

                # for categorical data, set get_feature_list
                if data_manifest.distribution == 'categorical':

                    # useful string constants derived from the sequence_type
                    vocab_list_name = '_'.join([sequence_type, 'vocab_list'])
                    vocab_file_path = subject.data_generator.sequence_type_to_vocab_file_path(
                        sequence_type)
                    vocab_pkl_path = os.path.join(
                        self.checkpoint_dir, '_'.join([sequence_type, 'vocab_file.pkl'])
                    )

                    self.vprint(
                        'Setting feature_list for %s to ' % data_key, end=''
                    )

                    # explicit vocab_list has priority 1
                    if vocab_list_name in kwargs:
                        self.vprint("argument passed w/key %s" % vocab_list_name)
                        class_list = kwargs[vocab_list_name]

                    # saved vocab_file has priority 2
                    elif vocab_file_path is not None:
                        self.vprint("vocab list stored in %s" % vocab_file_path)
                        class_list = subject.data_generator.get_class_list(
                            sequence_type
                        )

                    # a pickled vocab file has priority 3
                    elif os.path.isfile(vocab_pkl_path):
                        self.vprint("vocab list stored in %s" % vocab_pkl_path)
                        with open(vocab_pkl_path, 'rb') as fp:
                            bytes_list = pickle.load(fp)
                        class_list = [t.decode('utf-8') for t in bytes_list]

                    # none of the above, yet the data are still categorical
                    else:
                        self.vprint("training-intersection/validation-union")
                        special_tokens = (
                            [pad_token, EOS_token, OOV_token]
                            if 'sequence' in self._token_type
                            and 'encoder_' not in data_key
                            else [pad_token, OOV_token]
                        )
                        class_list = self._training_intersection_validation_union(
                            sequence_type, special_tokens
                        )

                    # and now set it (extremely verbosely because of python's
                    #  idiosyncratic late binding)
                    # data_manifest.get_feature_list = (
                    #     lambda class_list=class_list: class_list
                    # )
                    # work-around because lambdas can't be pickled
                    data_manifest.get_feature_list = partial(_identity, class_list)

                else:
                    # don't do anything for non-categorical data
                    pass

    @property
    def checkpoint_dir(self):

        # update the SequenceNetwork's checkpoint_path as well--if the net
        #  has been created at this point:
        try:
            self.net.checkpoint_path = os.path.join(
                self._checkpoint_dir, 'model.ckpt'
            )
        except AttributeError:
            pass
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
    def results_plotter(self):
        if self._results_plotter is None:
            subject = self.ecog_subjects[-1]
            self.results_plotter = plotters.ResultsPlotter(
                self.experiment_manifest[subject.subnet_id], subject,
                VERBOSE=self.VERBOSE, **self._RP_kwargs
            )

        return self._results_plotter

    @results_plotter.setter
    def results_plotter(self, results_plotter):
        # set up methods
        results_plotter.get_saliencies = self.get_saliencies
        results_plotter.get_encoder_embedding = self.get_encoder_embedding
        results_plotter.get_internal_activations = self.get_internal_activations

        self._results_plotter = results_plotter

    def torch_learn(self):
        import torch
        # somewhat hacky way to shoehorn PyTorch version in here...

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        ########
        # manifest from final subject only??
        torch_trainer = SequenceTrainer(
            self.experiment_manifest[self.ecog_subjects[-1].subnet_id],
            self.ecog_subjects,
            **self.ST_kwargs,
            REPORT_TRAINING_LOSS=True,
        )
        ########

        # something of a hack here for multi_trainers
        assessments = torch_trainer.train_and_assess(
            self.N_epochs, self.net, device
        )

        return assessments

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
        assessments = self.net.fit(self.ecog_subjects, **dict(fit_kwargs))
        self._save_results(assessments)

        # to facilitate restoring/assessing, update hard-coded restore_epochs
        if self._restore_epoch is not None:
            self.restore_epoch = (
                self.restore_epoch + self.net.N_epochs if RESUME else self.net.N_epochs
            )

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
                self.net.N_epochs = pretraining_epochs
                fit_kwargs['train_vars_scope'] = proprietary_scopes
                fit_kwargs['reuse_vars_scope'] = reusable_scopes
                fit_kwargs['_restore_epoch'] = latest_epoch
                self.net.fit([subject], **fit_kwargs)

                # then set up for next next training phase
                latest_epoch += self.net.N_epochs
                fit_kwargs['_restore_epoch'] = latest_epoch
                fit_kwargs['reuse_vars_scope'] = 'seq2seq'

            # full training
            if subject == self.ecog_subjects[-1]:
                training_epochs += posttraining_epochs
            self.net.N_epochs = training_epochs
            fit_kwargs['train_vars_scope'] = 'seq2seq'
            assessments = self.net.fit([subject], **fit_kwargs)
            latest_epoch += self.net.N_epochs
            self._save_results(assessments)

        # to facilitate restoring and assessing, store this
        self.restore_epoch = latest_epoch

        return assessments

    def assess_saved_model(self):

        self.update_net_from_saved_model()
        assessment_dict = self.net.restore_and_assess(
            self.ecog_subjects, self.restore_epoch)
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
        self, sequence_type, special_tokens=[]
    ):
        '''
        Typically used when neither a vocab_list nor a vocab_file has been
        provided, and not vocab_file.pkl has been found.
        '''

        # to get the class_list...
        targets_list = list(reduce(
            # ...reduce via the union across the DATA_PARTITIONS...
            lambda A, B: A | B, [
                reduce(
                    # ...of the reductions across the intersection or union...
                    (lambda A, B: A & B) if data_partition == 'training' else (
                        lambda A, B: A | B),
                    # ...of the class_list of this data_partition
                    [
                        set(s.write_tf_records_maybe(
                            sequence_type, [data_partition]
                        )) for s in self.ecog_subjects
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
            self.checkpoint_dir, 'model.ckpt') + '-%i' % self.restore_epoch)
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
                if re.match(r'subnet_\d*', subsubnet):
                    subnet_id = subsubnet.split('_')[1]
                    subsubnet = name_scopes.pop(0)
                else:
                    subnet_id = None

                # check if it's an RNN
                for scope in name_scopes:
                    match_obj = re.match(r'cell_\d*', scope)
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
                        data_sizes[subnet_id][subsubnet.replace('_projection', '_targets')] = \
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

        # In SequenceNets, the encoder RNN is constructed in a python loop,
        #  rather than within tf function, so it is the 'encoder_rnn' scope
        #  that gets numbered, rather than the cells.  Here you convert all
        #  these 'encoder_rnn_n` keys to a single key, 'encoder_rnn'.
        encoder_rnn_sizes = []
        for layer_name, layer_size in sorted(layer_sizes.items()):
            if layer_name.startswith('encoder_rnn'):
                encoder_rnn_sizes += layer_size
                layer_sizes.pop(layer_name)
        layer_sizes['encoder_rnn'] = encoder_rnn_sizes

        return layer_sizes, data_sizes, encoder_strides, EMA

    def _save_results(self, assessments):
        '''
        Write out to a text file
        '''

        # the save-file path/name
        subject = self.ecog_subjects[-1]
        experiment_manifest = self.experiment_manifest[subject.subnet_id]
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
                    '-'.join(str(N) for N in sizes)
                    for key, sizes in sorted(self.net.layer_sizes.items())
                ]
            )
        )
        print('save file is ' + save_file_path)

        # variables used for for plotting
        plot_interval = self.net.assessment_epoch_interval
        max_epoch = len(assessments['training'].decoder_accuracies)*plot_interval
        accuracies_epochs = [epoch for epoch in range(0, max_epoch, plot_interval)]

        # save the accuracies to a text file
        np.savetxt(
            save_file_path,
            np.stack([
                assessments['training'].decoder_accuracies,
                assessments['training'].decoder_word_error_rates,
                assessments['validation'].decoder_accuracies,
                assessments['validation'].decoder_word_error_rates,
                np.array(accuracies_epochs)
            ], axis=1),
            fmt="%.4f",
            header=(
                'training accs | training WERs | '
                'validation acc | validation WERs | epochs'
            )
        )

        # confusion matrix looks bad in tensorboard, so rebuild here
        decoder_targets_list = subject.data_manifests[
            'decoder_targets'].get_feature_list()
        N = subject.data_manifests['decoder_targets'].num_features
        if N < 100:
            fig_dimension = N//6
            confusions = assessments['validation'].decoder_confusions
            if confusions is not None:
                fig = heatmap_confusions(
                    plt.figure(figsize=(fig_dimension, fig_dimension)),
                    confusions,
                    x_axis_labels=decoder_targets_list,
                    y_axis_labels=decoder_targets_list,
                )
                fig.savefig(os.path.join(
                    save_file_dir, '%s_confusions.pdf' % self._token_type),
                    bbox_inches='tight')

    def count_all_targets(self, data_key='decoder_targets', threshold=0.4):

        # which targets do you want to count?
        targets_list = self.ecog_subjects[-1].data_manifests[
            data_key].get_feature_list()

        # dump into two tuples (each entry in a tuple corresponds to a subject)
        target_counters, sequence_counters = zip(*[
            subj.count_targets(targets_list, threshold)
            for subj in self.ecog_subjects
        ])

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
            # 'vocab_file',
        }

        params_series = [pd.Series(
            {
                # **{k: v for k, v in s.__dict__.items() if not k.startswith('_')},
                **{key: getattr(manifest, 'num_features')
                   for key, manifest in s.data_manifests.items()},
                **{'_'.join([manifest.sequence_type, 'vocab_list']):
                    manifest.get_feature_list()
                    for manifest in s.data_manifests.values()
                    if manifest.distribution == 'categorical'},
                **{attr: getattr(s, attr) for attr in subject_attributes},
                **{attr: getattr(self, attr) for attr in trainer_attributes},
            },
            name=s.subnet_id) for s in self.ecog_subjects
        ]
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
            [subject], self.restore_epoch,
            data_partition='validation', assessment_type=assessment_type
        )

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
                self.net._generate_oneshot_datasets(subnet_params, 0)
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
                ####
                # HARD-CODED for 'encoder_1_targets'
                _, get_targets_lengths = nn.sequences_tools(
                    GPU_op_dict['encoder_1_targets'])
                reverse_targets = tf.reverse_sequence(
                    GPU_op_dict['encoder_1_targets'], get_targets_lengths,
                    seq_axis=1, batch_axis=0)
                decimate_reversed_targets = reverse_targets[
                    :, 0::subnet_params.decimation_factor, :]
                ####

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
                _, get_final_state, _, _ = self.net._encode_sequences(
                    GPU_op_dict, subnet_params, 0.0, 0.0, set_initial_ind=0,
                )

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

    def tf_record_to_numpy_data(self, subj_id, block_id):
        '''
        It is frequently useful to inspect the content of the tf_records.

        NB: this method *does* reshape flattened ECoG data, but does *not*
        substitute indices for strings.

        USAGE: 
            for example in trainer.tf_record_to_numpy_data(401, 4):
                print(example.keys())
        '''

        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.reset_default_graph()

        # get the requested ECoGSubject
        for subject in self.ecog_subjects:
            if subject.subj_id == subj_id:
                break
        else:
            raise ValueError('Requested subject not in this trainer')

        # block default transforms, e.g. of strings to indices; see subjects.py
        None_transforms = []
        for key, data_manifest in subject.data_manifests.items():
            if data_manifest._transform is None:
                None_transforms.append(key)
                subject.data_manifests[key]._transform = lambda seq: seq

        # pull the tf_record into a TF dataset
        dataset = tf.data.TFRecordDataset(
            [subject.tf_record_partial_path.format(block_id)]
        )

        # parse according to the info in the data_manifests
        dataset = dataset.map(
            lambda example_proto: tfh.parse_protobuf_seq2seq_example(
                example_proto, subject.data_manifests
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        # remove transform blocking by restoring original transforms
        for key in None_transforms:
            subject.data_manifests[key]._transform = None

        # set up the one-shot iterator
        iterator = tf.compat.v1.data.Iterator.from_structure(
            tf.compat.v1.data.get_output_types(dataset),
            tf.compat.v1.data.get_output_shapes(dataset)
        )
        initializer = iterator.make_initializer(dataset)
        sequenced_op_dict = iterator.get_next()

        # finally, transform to numpy data
        with tf.compat.v1.Session() as sess:
            sess.run(initializer)
            while True:
                try:
                    yield sess.run(sequenced_op_dict)
                except tf.errors.OutOfRangeError:

                    # bring back eager execution, or other things will break
                    tf.compat.v1.enable_eager_execution()
                    break


def construct_online_predictor(
    restore_dir, targets_list=None, TARGETS_ARE_SEQUENCES=False
):

    # open a session with the saved_model loaded into it
    sess = tfh.get_session_with_saved_model(restore_dir)

    # create a function which uses this session to decode
    def predict(inputs):
        decoded_probs, sequenced_decoder_outputs = sess.run(
            ['decoder_probs:0', 'decoder_outputs:0'],
            feed_dict={'encoder_inputs:0': inputs}
        )
        if targets_list:
            tokens_list = (
                targets_list if TARGETS_ARE_SEQUENCES else
                nn.targets_to_tokens(targets_list, pad_token)
            )
            hypotheses = target_inds_to_sequences(
                sequenced_decoder_outputs, tokens_list)[0]
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


def _identity(x):
    return x
