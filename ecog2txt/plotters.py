# standard libraries
import os
import pdb
import sys
import re
from functools import reduce
import json
import copy
import itertools
from collections import Counter
from IPython.display import clear_output
from cycler import cycler

# third-party packages
import numpy as np
import hickle
from scipy.stats import t as students_t
from scipy.stats import wilcoxon
from scipy.io import loadmat
import seaborn as sns
import pandas as pd
from tensorflow.compat.v1.python_io import tf_record_iterator
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
if os.name == 'posix' and "DISPLAY" not in os.environ:
    mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# local
from machine_learning.neural_networks import tf_helpers as tfh
from utils_jgm.tikz_pgf_helpers import tpl_save
from utils_jgm.toolbox import cubehelix2params, pseudomode, wer_vector
from utils_jgm.toolbox import auto_attribute, str2int_hook
from utils_jgm.toolbox import barplot_annotate_brackets
from utils_jgm.toolbox import anti_alias
from ecog2txt import subjects as e2t_subjects
import ecog2txt

mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
#mpl.rcParams['text.usetex'] = True

##############
# TO DO:
# (1) restore functionality for USE_FIELD_POTENTIALS
# (2) restore functionality for input_mask
# (3) restore occlusion training
##############


class DecodingResults:
    @auto_attribute
    def __init__(
        self,
        decoding_results_file_name,
        VERBOSE
    ):

        if os.path.isfile(decoding_results_file_name):
            self.vprint('Found decoding results; loading into attributes...')
            try:
                hickled_data = hickle.load(decoding_results_file_name)
            except BaseException as e:  # ModuleNotFoundError:
                # HACK for backward compatibility with old code structure:
                #  a function may have been saved in the hickle file.
                sys.modules['pycode.ecog2txt'] = ecog2txt
                ########
                # A hack for a hack--required for later versions of sys...
                sys.modules['pycode'] = ecog2txt
                ########
                hickled_data = hickle.load(decoding_results_file_name)

            # for backward compatibility with old saved results
            if type(hickled_data) is tuple:
                hickled_data = hickled_data[0]
                
            # for some results, you saved a list of the training_blocks sets
            blocks = np.array(hickled_data['training_blocks'])
            self.Ndatasizes = len({len(blks) for blks in blocks})
            self.training_blocks = np.reshape(blocks, (-1, self.Ndatasizes))

            # for most results, you saved a list of the validation_blocks
            blocks = np.array(hickled_data['validation_blocks'])
            ############
            # Is this still necessary??
            if type(blocks[0]) is set:
                self.validation_blocks = np.reshape(
                    blocks, (-1, self.Ndatasizes))
            else:
                self.validation_blocks = np.reshape(
                    blocks, (-1, self.Ndatasizes, blocks.shape[-1]))
            ############

            # all results in *percent*
            for result in ['word_error_rate', 'accuracy', 'nums_seconds']:
                data = hickled_data[result]
                data = data[:, -1] if len(data.shape) == 2 else data
                setattr(self, result, ResultsMatrix(np.reshape(
                    data, (-1, self.Ndatasizes))))

        else:
            self.vprint('No decoding results found at ', end='')
            self.vprint('%s!; ' % decoding_results_file_name, end='')
            self.vprint('setting attributes to None')
            for attribute in {
                'Ndatasizes', 'training_blocks', 'validation_blocks',
                'word_error_rate', 'accuracy', 'nums_seconds'
            }:
                setattr(self, attribute, None)

        #######
        # DEPRECATED
        self.all_training_blocks = None,
        self.nums_training_blocks = None,
        #######

    def vprint(self, *args, **kwargs):
        if self.VERBOSE:
            print(*args, **kwargs)


class ResultsPlotter():
    @auto_attribute(CHECK_MANIFEST=True)
    def __init__(
        self,
        manifest,
        subject,
        suffix='',
        contrib_method='decoder_saliency_map',
        line_style='solid',
        #####
        # in the manifest
        token_type=None,
        subject_name=None,
        alias=None,
        elevation=None,
        azimuth=None,
        RGB_color=None,
        num_unique_training_sentences=None,
        saved_results_dir=None,
        electrode_path=None,
        tf_record_partial_path=None,
        tikz_partial_path=None,
        png_partial_path=None,
        anatomy_grand_list=None,
        grid_names=None,
        #####
        VERBOSE=True,
    ):

        # LOAD OCCLUSION-TRAINED DECODING RESULTS
        # occlusion_results_file_name = os.path.join(
        #     self.saved_results_dir,
        #     'occlusion_sensitivity_{0}_{1}.hkl'
        # ).format(self.subject.subnet_id, experiment)
        # if os.path.isfile(occlusion_results_file_name):
        #     self.vprint('Found occlusion results; loading into attributes...')
        #     hickled_data = hickle.load(occlusion_results_file_name)
        #     self.masked_start_electrodes = hickled_data[2]
        #     self.masked_word_error_rates = hickled_data[0]['word_error_rate']
        # else:
        #     self.vprint('No occlusion-training results found!')
        #     for attribute in [
        #         'masked_start_electrodes', 'masked_word_error_rates',
        #     ]:
        #         setattr(self, attribute, None)

        # ...
        self._testtime_occlusion_contributions = None

        # engage some setters
        self.elec_contribs = None
        self.anatomy_labels = None
        self.elec_contrib_sequences = None

        # initialize
        self._decoding_results = None
        self._nums_nominal_repeats = None
        self._nums_counted_repeats = None

    @property
    def saved_results_dir(self):
        return '.' if self._saved_results_dir is None else self._saved_results_dir

    @saved_results_dir.setter
    def saved_results_dir(self, saved_results_dir):
        if hasattr(self, '_saved_results_dir'):
            # check if the argument works as a subdir of its predecessor
            full_dir = os.path.join(self._saved_results_dir, saved_results_dir)
            if os.path.isdir(full_dir):
                if self.VERBOSE:
                    print("INTERPRETING USER-PROVIDED SAVED_RESULTS_DIR AS SUBDIR")
                self._saved_results_dir = full_dir
                return

        # ok, it's the whole path
        if self.VERBOSE:
            print("USING USER-PROVIDED SAVED_RESULTS_DIR")
        self._saved_results_dir = saved_results_dir

    @property
    def decoding_results(self):
        if self._decoding_results is None:
            decoding_results_file_name = os.path.join(
                self.saved_results_dir, 'perf_vs_training_size_{0}_{1}.hkl'
            ).format(self.subject.subnet_id, self.suffix)
            self._decoding_results = DecodingResults(
                decoding_results_file_name, self.VERBOSE)

        return self._decoding_results

    @decoding_results.setter
    def decoding_results(self, decoding_results):
        self._decoding_results = decoding_results

    @property
    def Ndatasizes(self):
        return self.decoding_results.Ndatasizes

    @property
    def training_blocks(self):
        return self.decoding_results.training_blocks

    @property
    def validation_blocks(self):
        return self.decoding_results.validation_blocks

    @property
    def word_error_rate(self):
        return self.decoding_results.word_error_rate

    @property
    def accuracy(self):
        return self.decoding_results.accuracy

    @property
    def nums_seconds(self):
        return self.decoding_results.nums_seconds

    def vprint(self, *args, **kwargs):
        if self.VERBOSE:
            print(*args, **kwargs)

    @property
    def RGB_color(self):
        return [0, 0, 0] if self._RGB_color is None else self._RGB_color

    @RGB_color.setter
    def RGB_color(self, RGB_color):
        self._RGB_color = RGB_color

    @property
    def elevation(self):
        return 0 if self._elevation is None else self._elevation

    @elevation.setter
    def elevation(self, elevation):
        self._elevation = elevation

    @property
    def azimuth(self):
        return 0 if self._azimuth is None else self._azimuth

    @azimuth.setter
    def azimuth(self, azimuth):
        self._azimuth = azimuth

    @property
    def nums_nominal_repeats(self):
        if self._nums_nominal_repeats is None:
            print('counting repeats', end='')
            Nfolds = self.validation_blocks.shape[0]
            num_repeats = np.zeros((Nfolds, self.Ndatasizes))
            for iRepeat in range(Nfolds):
                for iNblocks, num_blocks in enumerate(self.nums_training_blocks):
                    for block in self.all_training_blocks[iRepeat, :num_blocks]:
                        num_repeats[iRepeat, iNblocks] += sum(
                            1 for _ in tf_record_iterator(
                                self.tf_record_partial_path.format(
                                    block)))
                print('.', end='')
            print('.')

            if self.num_unique_training_sentences is None:
                raise ValueError(
                    'nums_nominal_repeats doesn''t work w/this experiment type'
                )
            else:
                self._nums_nominal_repeats = ResultsMatrix(
                    num_repeats/self.num_unique_training_sentences)

        return self._nums_nominal_repeats

    @property
    def nums_counted_repeats(self):

        # do *not* transform the saved strings into indices!
        target_manifest = copy.copy(
            self.subject.data_manifests['decoder_targets'])
        target_manifest.transform = None

        if self._nums_counted_repeats is None:
            print('counting repeats', end='')
            Nfolds = self.validation_blocks.shape[0]
            num_repeats = np.zeros((Nfolds, self.Ndatasizes))

            for iRepeat in range(Nfolds):
                for iNblocks in range(self.Ndatasizes):

                    # for task transfer learning
                    training_blocks = (
                        self.task_training_blocks[iRepeat, iNblocks]
                        if self.task_training_blocks is not None else
                        self.training_blocks[iRepeat, iNblocks]
                    )

                    # count
                    sequence_counter = e2t_subjects.SequenceCounter(())
                    e2t_subjects.apply_to_all_tf_examples(
                        [sequence_counter],
                        lambda example_proto: tfh.parse_protobuf_seq2seq_example(
                            example_proto, {'decoder_targets': target_manifest}
                        ),
                        training_blocks, self.tf_record_partial_path
                    )
                    num_repeats[iRepeat, iNblocks] = pseudomode(
                        sequence_counter.types)
                print('.', end='')
            print('.')
            self._nums_counted_repeats = ResultsMatrix(num_repeats)

        return self._nums_counted_repeats

    @property
    def ordered_good_electrodes(self):
        '''
        The canonical ordering scheme for all ResultsPlotters, it is inherited
         from the order of the data_generator.good_channels and hence
         guaranteed to line up with the weight matrices--unless the calculation
         in data_generator was changed between the times the tf_records were
         created and this file is run, in which case the the plots will be
         erroneous.  To be safe, one could always (1) rewrite the tf_records,
         (2) train a new network on them, and then (3) run the plotters.
        '''

        # shorthand
        good_channels = self.subject.data_generator.good_channels

        # in all cases the object is a matrix (2-dimensional)
        if self.subject.data_generator.REFERENCE_BIPOLAR:
            return self.subject.data_generator.bipolar_to_elec_map[
                good_channels, :]
        else:
            return np.array(good_channels)[:, None]

    @property
    def anatomy_labels(self):

        if self._anatomy_labels is not None:
            return self._anatomy_labels
        else:
            if not os.path.isfile(self.electrode_path):
                self.vprint('No elec data found; setting ', end='')
                self.vprint('anatomy_labels to first anatomical area')
                return self.ordered_good_electrodes.shape[0]*[
                    self.anatomy_grand_list[0]]
            else:
                # For each grid_name, construct a map from electrode number to
                #  anatomical label.
                electrode_data = loadmat(self.electrode_path)
                electrode_to_anatomy_maps = [{
                    int(re.findall(r'\d+', label[1][0])[-1]) - 1: anat[3][0]
                    for label, anat in zip(
                        electrode_data['eleclabels'], electrode_data['anatomy'])
                    if str(label[1][0]).startswith(grid_name)
                } for grid_name in self.grid_names]

                # combine the maps, augmenting nominal electrode number by the
                #  number of electrodes already in the combined map.
                electrode_to_anatomy_map = reduce(
                    lambda x, y: {**x, **{key+len(x): value for key, value in y.items()}},
                    electrode_to_anatomy_maps
                )

                # Unfortunately, the stored anatomy labels for some subjects
                #  are defined wrt a *standard* grid.  We have to adjust for
                #  that here.  If the elec_layout matches the std_elec_layout,
                #  this will just assign the self.ordered_good_electrodes[:, 0]
                #  to electrodes.
                elec_layout = self.subject.data_generator.elec_layout
                grid_size = self.subject.data_generator.grid_size
                std_elec_layout = np.arange(np.prod(
                    grid_size)-1, -1, -1).reshape(grid_size).T
                electrodes = [
                    std_elec_layout[np.where(elec_layout == e)][0]
                    for e in self.ordered_good_electrodes[:, 0]
                ]

                # get the anatomy labels
                anatomy_labels = [electrode_to_anatomy_map[e] for e in electrodes]

                # pool certain areas?
                pooling_dict = {
                    'middle frontal': {'rostralmiddlefrontal', 'caudalmiddlefrontal'},
                    'IFG': {'parstriangularis', 'parsopercularis', 'parsorbitalis'},
                    'vSMC': {'postcentral', 'precentral'}
                }
                for pooled_area, poolable_areas in pooling_dict.items():
                    anatomy_labels = [
                        pooled_area.replace(' ', '') if label in poolable_areas and
                        pooled_area in self.anatomy_grand_list else label
                        for label in anatomy_labels
                    ]

                # convert to versions with spaces in the names
                oneword_areas = [
                    area.replace(' ', '') for area in self.anatomy_grand_list]
                assert set(anatomy_labels) <= set(oneword_areas), 'Missed an area!'
                return [
                    self.anatomy_grand_list[oneword_areas.index(label)]
                    for label in anatomy_labels
                ]

    @anatomy_labels.setter
    def anatomy_labels(self, anatomy_labels):
        self._anatomy_labels = anatomy_labels

    @property
    def electrode_locs_2D(self):

        # flipud to convert from matrix inds ("ij") to image coords ("xy")
        img_layout = np.flipud(self.subject.data_generator.elec_layout)

        _, _, Y, X = (self.ordered_good_electrodes[:, :, None, None] ==
                      img_layout[None, None, :, :]).nonzero()
        x = np.mean(X.reshape(self.ordered_good_electrodes.shape), 1)
        y = np.mean(Y.reshape(self.ordered_good_electrodes.shape), 1)

        # return a matrix of size N_good_electrodes x 2
        return np.stack((x, y)).T

    @property
    def electrode_locs_3D(self):
        if not os.path.isfile(self.electrode_path):
            self.vprint('No elec data found; setting electrode_locs_3D to None')
            return None
        else:
            # when given more than one electrode at once, *average* their locs
            all_electrode_locs = loadmat(self.electrode_path)['elecmatrix']
            return np.mean(all_electrode_locs[self.ordered_good_electrodes, :],
                           axis=1)

    @property
    def hemisphere(self):
        if not os.path.isfile(self.electrode_path):
            self.vprint('No elec data found; setting hemisphere to None')
            return None
        else:
            if all(self.electrode_locs_3D[:, 0] > 0):
                return 'rh'
            elif all(self.electrode_locs_3D[:, 0] < 0):
                return 'lh'
            else:
                raise ValueError("Unexpected electrode locations!")

    @property
    def elec_contribs(self):
        if self._elec_contribs is not None:
            contributions = self._elec_contribs
        else:
            if self.contrib_method == 'weight_norms':
                contributions = self.weight_norms
            elif 'saliency_map' in self.contrib_method:
                contributions = self.get_saliencies(self.contrib_method)
            elif self.contrib_method == 'occlusion_trained':
                contributions = self.traintime_occlusion_contributions
            elif self.contrib_method == 'occlusion_tested':
                contributions = self.testtime_occlusion_contributions
            else:
                raise ValueError("Unknown self.contrib_method")

            # This sets the shadow property, _elec_contribs.  NB that it is
            #  *not* normalized into [0, 1].  This allows the normalization to
            #  be applied to data assigned externally with the setter.
            self.elec_contribs = contributions

        # convert from absolute to relative contributions (above baseline)
        #  and put into [0, 1] range.
        contributions -= np.min(contributions)
        contributions /= np.max(contributions)

        return contributions

    @elec_contribs.setter
    def elec_contribs(self, elec_contribs):
        self._elec_contribs = elec_contribs

    @property
    def contrib_method(self):
        return self._contrib_method

    @contrib_method.setter
    def contrib_method(self, method):
        self._contrib_method = method
        self._elec_contribs = None   # (re)set this

    @property
    def weight_norms(self):

        W = self.get_encoder_embedding()
        if len(W.shape) == 4:
            self.vprint('found conv. weights; averaging across temporal dimension')
            # so W.shape = (1, filter_width, num_channels, num_units)
            return np.linalg.norm(W[0], ord='fro', axis=(0, 2))
        else:
            return np.linalg.norm(W, ord=2, axis=1)

    def flat_conv_embedding(self):
        # retrieve and then line up the conv matrices next to each other

        # W: (1 x T x Nelectrodes x Nunits)
        W = self.get_encoder_embedding()

        '''
        W = W.transpose([0, 2, 1, 3]).reshape(
            W.shape[0]*W.shape[2], W.shape[1]*W.shape[3])
        '''
        # W: (T*Nelectrodes x Nunits)
        W = W.reshape(W.shape[1]*W.shape[2], W.shape[3])

        return W

    @property
    def elec_contrib_sequences(self):
        if self._elec_contrib_sequences is not None:
            elec_contrib_sequences = self._elec_contrib_sequences
        else:
            elec_contrib_sequences, _ = self.get_saliencies(
                self.contrib_method, 'sequences')

            #########
            # This anti-aliasing introduces a big spike at the beginning.
            #  Consider changing to the one in ecogVIS....
            dfactor = self.subject.decimation_factor
            f_decimated = self.subject.data_generator.sampling_rate/dfactor
            f_Nyquist = f_decimated/2

            for elec_contrib_sequence in elec_contrib_sequences:
                anti_alias(
                    elec_contrib_sequence,
                    self.subject.data_generator.sampling_rate,
                    f_Nyquist, 0.2*f_Nyquist, atten_DB=40,
                )
            #########
            self.elec_contrib_sequences = elec_contrib_sequences

        #######
        # here you can make any final changes....
        #######
        return elec_contrib_sequences

    @elec_contrib_sequences.setter
    def elec_contrib_sequences(self, elec_contrib_sequences):
        self._elec_contrib_sequences = elec_contrib_sequences

    @property
    def traintime_occlusion_contributions(self):

        # for restoring after we're done (it's probably None)
        old_input_mask = copy.copy(self.subject.input_mask)

        # create a generic 2x2 input mask
        subgrid_size = [2, 2]  # HARD-CODED!
        self.subject.input_mask = e2t_subjects.SubgridParams(
            grid_size=self.grid_size, subgrid_size=subgrid_size, start=[0, 0],
            SUBSAMPLE=False, OCCLUDE=False)

        # init
        WERs_list = [
            [] for ind in range(self.subject.data_generator.num_ECoG_channels)
        ]

        # for all WERs, collect up the corresponding masked electrodes
        for start, wer in zip(
            self.masked_start_electrodes, self.masked_word_error_rates
        ):
            self.subject.input_mask.start = start
            for ind in self.subject.input_mask.subgrid_inds:
                WERs_list[ind].append(wer)

        # prevent the input mask from messing up other analyses
        self.subject.input_mask = old_input_mask

        # (occluded electrodes contribution) ~ <WER>, i.e. *increases* in WER
        # indicate important electrodes
        contribs = np.array([np.nanmean(wers_list) for wers_list in WERs_list])
        contribs[np.isnan(contribs)] = np.nanmin(contribs)
        return contribs

    ######################
    # BROKEN
    @property
    def testtime_occlusion_contributions(self):
        if self._testtime_occlusion_contributions is None:

            # this is probably None
            old_inputs_to_occlude = self.trainer.net.inputs_to_occlude

            # create a generic 2x2 input mask
            subgrid_size = [2, 2]  # HARD-CODED!
            input_mask = e2t_subjects.SubgridParams(
                grid_size=self.grid_size, subgrid_size=subgrid_size,
                start=[0, 0], SUBSAMPLE=False, OCCLUDE=False)
            input_mask.subj_id = self.subject.subnet_id
            input_mask.good_channels = self.subject.data_generator.good_channels

            # malloc
            WERs_list = [[] for ind in range(
                self.subject.data_manifests['encoder_inputs'].num_features)
            ]

            # for each possible mask location...
            for start_i in range(0, self.grid_size[0] - subgrid_size[0]):
                clear_output(wait=True)
                print('STARTING START_i = %i' % start_i)
                for start_j in range(0, self.grid_size[1] - subgrid_size[1]):

                    # update the location of the mask and assess
                    input_mask.start = [start_i, start_j]
                    self.trainer.net.inputs_to_occlude = input_mask.subgrid_inds
                    assessments = self.trainer.net.restore_and_assess(
                        self.trainer.ecog_subjects, self.trainer.restore_epoch,
                        WRITE=False)

                    for ind in input_mask.subgrid_inds:
                        WERs_list[ind].append(assessments['validation'].word_error_rate)

            # restore this
            self.trainer.net.inputs_to_occlude = old_inputs_to_occlude

            # ...
            contribs = np.array([np.nanmean(wers_list) for wers_list in WERs_list])
            contribs[np.isnan(contribs)] = np.nanmin(contribs)
            self._testtime_occlusion_contributions = contribs

        return self._testtime_occlusion_contributions
    ######################

    @property
    def task_training_blocks(self):
        if (
            suffix_to_label(self.suffix) in ['+dual TL', '+task TL'] and
            self.training_blocks is not None and
            self.validation_blocks is not None
        ):
            # shorthand
            block_dict = self.subject._block_dict

            # for every model that was trained...
            task_training_blocks = []
            for t_blocks, v_blocks in zip(
                self.training_blocks.flatten(),
                self.validation_blocks.flatten()
            ):
                # get the validation block *types*
                validation_block_types = {
                    block_dict[block]['type'] for block in v_blocks
                }

                # assemble training blocks w/types among validation_block_types
                task_training_blocks.append({
                    block for block in t_blocks
                    if block_dict[block]['type'] in validation_block_types
                })
            return np.reshape(
                task_training_blocks, self.training_blocks.shape)
        else:
            return None

    def line_plot_performance_vs_amount_of_training_data(
        self, performance_measure, versus, fig_num,
        x_major_ticks=np.arange(0, 41, 10), y_major_ticks=np.arange(6)/5,
        ymin=0.0, ymax=100.0, legend_entry=None,
    ):

        ####
        # TO DO:
        # (1) get rid of outline

        # ordinate
        perf_means = 100*getattr(self, performance_measure).mean
        perf_std_errors = 100*getattr(self, performance_measure).std_err

        # abscissa
        if versus == 'minutes':
            data_amount = ResultsMatrix(self.nums_seconds.data/60)
            x_label = 'minutes of training data'
        elif versus == 'nominal repeats':
            # slightly less reliable...
            data_amount = self.nums_nominal_repeats
            x_label = 'number of training repeats'
        elif versus == 'counted repeats':
            data_amount = self.nums_counted_repeats
            x_label = 'number of training repeats'
        else:
            raise ValueError("Unexpected variable to plot! - jgm")

        # plot
        fig = plt.figure(fig_num)
        ax = fig.get_axes()[0] if fig.get_axes() else fig.add_subplot(111)
        p = ax.errorbar(
            data_amount.mean, perf_means,
            xerr=data_amount.std_err, yerr=perf_std_errors,
            color=self.RGB_color, linestyle=self.line_style, linewidth=1.0,
            label=legend_entry
        )

        # label things
        xmin, xmax = ax.get_xlim()
        ax.set_xlim((0, xmax))
        ax.set_ylim((ymin, ymax))
        ax.grid(visible=True, color='black')
        ax.set_xticks(x_major_ticks)
        ax.set_yticks(y_major_ticks)
        ax.set_xlabel(x_label)
        ax.set_ylabel(performance_measure.replace('_', ' ') + ' (\%)')

        #ax.spines['bottom'].set_color('1.0')
        #ax.spines['top'].set_color('1.0')
        #ax.spines['right'].set_color('1.0')
        #ax.spines['left'].set_color('1.0')

        return p

    def scatter_electrode_contributions(
        self, plot_style='no brain', LABEL=False, SAVE=True, suffix='',
        axis=None, max_marker_size=None
    ):
        from palettable.colorbrewer import qualitative

        # via trial and error...
        style_to_max_marker_map = {
            'on brain': 0.3,  # 0.2
            'no brain': 35,
            'flat': 650
        }
        if max_marker_size is None:
            max_marker_size = style_to_max_marker_map[plot_style]

        # save as:
        figure_name = 'electrode_locations' + suffix

        # set the colors
        elec_color_scheme = getattr(
            qualitative, 'Set3_%i' % len(self.anatomy_grand_list))
        cmap = elec_color_scheme.mpl_colormap
        good_color_ids = [
            self.anatomy_grand_list.index(label)
            for label in self.anatomy_labels
        ]

        # set the legend
        unique_color_ids = set(good_color_ids)
        legend_colors = [cmap(color_id/(len(self.anatomy_grand_list)-1))
                         for color_id in unique_color_ids]
        legend_labels = [self.anatomy_grand_list[i] for i in unique_color_ids]

        def scatter_on_brain():

            from img_pipe import img_pipe
            patient = img_pipe.freeCoG(subj=self.subject_name, hem=self.hemisphere)
            #####
            # If VTK makes the *radii* of the spheres proportional to the norm
            #  of the marker_sizes, then taking a square root first makes the
            #  radii proportional to the square root fof the weight norms. This
            #  in turn makes the *areas* of the disks in 2D proportional to the
            #  weight norms.
            #####
            marker_sizes = np.tile(self.elec_contribs[:, None], 3)
            mesh, mlab = patient.plot_recon_anatomy(
                showfig=True, screenshot=True, show_numbers=False,
                marker_size=max_marker_size*marker_sizes**(1/2),
                elecfile_prefix=None, SHOW_TITLE=False, save_dir='',
                good_labels=legend_labels, color_list=legend_colors,
                elec_colors=np.array([
                    np.array(cmap(i/(len(self.anatomy_grand_list)-1))[:3])
                    for i in good_color_ids]),
                elec_locs=self.electrode_locs_3D,
            )
            mlab.options.offscreen = True
            if SAVE:
                mlab.savefig(self.png_partial_path.format('anatomy'))

            return None, None

        def scatter_no_brain():
            # ...
            fig = plt.figure(figsize=(5, 5), dpi=150)
            ax = fig.gca(projection='3d')
            ax.scatter(
                *self.electrode_locs_3D.T,
                s=max_marker_size*self.elec_contribs,
                c=good_color_ids, cmap=cmap,
                vmin=0, vmax=len(self.anatomy_grand_list)-1,
                edgecolors='black', linewidths=0.5
            )
            ax.view_init(self.elevation, self.azimuth)

            if SAVE:
                tpl_save(filepath=self.tikz_partial_path.format(figure_name))
                fig.savefig(self.png_partial_path.format(figure_name))

            # label the electrodes with their numbers?
            if LABEL:
                from mpl_toolkits.mplot3d import proj3d
                for i, electrode_loc in enumerate(self.electrode_locs_3D):
                    x2, y2, _ = proj3d.proj_transform(
                        *electrode_loc, ax.get_proj())
                    ax.annotate(i, (x2, y2), fontsize=6)

            return fig, ax

        def scatter_flat():
            # scatter
            if axis is None:
                fig, ax = plt.subplots(
                    figsize=[s/2 for s in self.subject.data_generator.grid_size]
                )
            else:
                fig = axis.figure
                ax = axis
            ax.scatter(
                *self.electrode_locs_2D.T,
                s=max_marker_size*self.elec_contribs,
                c=good_color_ids, cmap=cmap,
                vmin=0, vmax=len(self.anatomy_grand_list)-1,
                edgecolors='black', linewidths=0.5
            )
            ax.axis('off')

            # make a legend
            custom_lines = [
                mpl.lines.Line2D([0], [0], lw=4, color=c) for c in legend_colors
            ]
            ax.legend(
                custom_lines, legend_labels,
                bbox_to_anchor=(0, -0.01), loc="upper left"
            )

            # label the electrodes with their numbers?
            if LABEL:
                for i, x_i, y_i in zip(
                    self.ordered_good_electrodes[:, 0],
                    *self.electrode_locs_2D.T
                ):
                    ax.annotate(i, (x_i, y_i), fontsize=12)

            # save
            if SAVE:
                tpl_save(
                    filepath=self.tikz_partial_path.format(figure_name),
                    extra_axis_parameters={'width=\\figwidth', 'height=\\figheight'},
                    extra_lines_start={
                        '\\providecommand{{\\figwidth}}{{{0}in}}%'.format(
                            self.subject.data_generator.grid_size[0]/2),
                        '\\providecommand{{\\figheight}}{{{0}in}}%'.format(
                            self.subject.data_generator.grid_size[1]/2)
                    },
                )
                fig.savefig(self.png_partial_path.format(figure_name), dpi=400)

            return fig, ax

        # ...
        style_to_fxn_map = {
            'on brain': scatter_on_brain,
            'no brain': scatter_no_brain,
            'flat': scatter_flat
        }

        return style_to_fxn_map[plot_style]()

    def animate_electrode_contributions(self, iExample=0):

        # this does make a copy, because you're fancy indexing
        elec_contrib_sequence = self.elec_contrib_sequences[
            iExample,
            np.nonzero(np.sum(np.abs(self.elec_contrib_sequences[iExample]),
                              axis=1))[0]
        ]
        ######
        # Kind of a hack.  The anti-aliasing (?) introduces a spike at zero, so
        #  we ignore that part of the sequence when normalizing
        elec_contrib_sequence -= np.min(elec_contrib_sequence[20:])
        elec_contrib_sequence /= np.max(elec_contrib_sequence[20:])
        ######
        anim = self.animate_electrode_activities(
            elec_contrib_sequence.T,
            [i for i in range(elec_contrib_sequence.shape[0])]
        )

        # TO RUN THIS IN JUPYTER, do:
        #   > from IPython.display import HTML
        #   > HTML(anim.to_jshtml())
        # You may also want to increase the available MB:
        #   > mpl.rcParams['animation.embed_limit'] = 60
        return anim

    def animate_electrode_activities(self, size_data, title_data):

        from palettable.colorbrewer import qualitative

        # sizes
        max_marker_size = 650

        # colors
        colors = [self.anatomy_grand_list.index(label)
                  for label in self.anatomy_labels]
        elec_color_scheme = getattr(
            qualitative, 'Set3_%i' % len(self.anatomy_grand_list))
        cmap = elec_color_scheme.mpl_colormap

        # create the figure
        fig, ax = plt.subplots(
            figsize=[s/2 for s in self.subject.data_generator.grid_size],
            dpi=40
        )
        ####
        # paths = ax.scatter([], [], lw=2)
        paths = ax.scatter(
            *self.electrode_locs_2D.T,
            s=max_marker_size*size_data[:, 0],
            c=colors, cmap=cmap,
            vmin=0, vmax=len(self.anatomy_grand_list)-1,
            edgecolors='black', linewidths=0.5
        )
        ####
        ax.axis('off')

        def init():
            paths.set_offsets(self.electrode_locs_2D)
            paths.set_sizes(max_marker_size*size_data[:, 0]),
            return (paths,)

        def animate(i):
            paths.set_sizes(max_marker_size*size_data[:, i])
            ax.set_title(
                title_data[i],
                # color=self.anatomical_colors[title_data[i]]
            )
            return (paths,)

        # set the interval (in ms) to achieve 1/4 speed
        dilation_factor = 4
        period_s = 1/self.subject.data_generator.sampling_rate
        period_ms = period_s*1000
        interval_ms = dilation_factor*period_ms

        return animation.FuncAnimation(
            fig, animate, init_func=init, frames=size_data.shape[1],
            interval=interval_ms, blit=True)

    def bar_plot_electrode_contributions(self, SHOW_Y_TICK_LABELS=True):

        # save as:
        plot_type = 'barplot'
        figure_name = '_'.join(['anatomical_contributions', plot_type])
        
        # parcel out the contributions by anatomical area
        boolean_indices = (np.array(self.anatomy_labels, ndmin=2) ==
                           np.array(self.anatomy_grand_list, ndmin=2).T)
        area_contributions = [self.elec_contribs[indarray]
                              for indarray in boolean_indices]

        # compute mean and std err, and then normalize so that means sum to 1.0
        average_contribution, std_err_contribution = [], []
        for contribution in area_contributions:
            if len(contribution) > 0:
                average_contribution.append(np.mean(contribution))
                std_err_contribution.append(
                    (np.var(contribution, ddof=1)/contribution.shape[0])**(1/2)
                )
            else:
                average_contribution.append(0)
                std_err_contribution.append(0)

        # plot
        fig, ax = plt.subplots()
        y_pos = range(len(self.anatomy_grand_list))
        #### Doesn't work with tikzplotlib so we need a workaround
        # ax.invert_yaxis()
        average_contribution = np.flip(average_contribution)
        std_err_contribution = np.flip(std_err_contribution)
        yticklabels = reversed(self.anatomy_grand_list)
        ####
        ax.barh(
            y_pos, average_contribution, xerr=std_err_contribution,
            color=self.RGB_color
        )
        ax.set_yticks(y_pos)
        ax.tick_params(axis='x', which='both', bottom=False, top=False,
                       labelbottom=False)

        # ax.set_xlabel('relative contribution (a.u.)')
        if SHOW_Y_TICK_LABELS:
            ax.set_yticklabels(yticklabels, rotation=0)
        else:
            ax.tick_params(axis='y', which='both', left=False, right=False,
                           labelleft=False)

        # export the figure to tikz
        tpl_save(
            filepath=self.tikz_partial_path.format(figure_name),
            extra_axis_parameters={
                'xticklabel style={/pgf/number format/fixed,/pgf/number format/precision=2}',
                'width=\\figwidth',
                'height=\\figheight'
            },
            extra_lines_start={
                '\\providecommand{\\figwidth}{360pt}%',
                '\\providecommand{\\figheight}{310pt}%'
            },
        )

    def kernel_density_plot_electrode_contributions(
        self, bw_adjust=1.0, y_upper_bound=35, label_color=None,
        VERTICAL=False
    ):

        # init
        plot_type = 'kdeplot'
        figure_name = '_'.join(['anatomical_contributions', plot_type])
        minor_contribution_areas = []
        sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

        # ...
        if VERTICAL:
            facet_kwargs = {
                'col_order': self.anatomy_grand_list,
                'col': 'areas',
                'aspect': 1/3,
                'height': 3.0
            }
            x_scatter_data = 'scatter_y'
            y_scatter_data = 'contributions'
            x_kde_data = None
            y_kde_data = 'contributions'
            ax_line = plt.axvline
            ax_line_kwargs = {'x': 0}
        else:
            facet_kwargs = {
                'row_order': self.anatomy_grand_list,
                'row': 'areas',
                'aspect': 6,
                'height': 0.75
            }
            x_scatter_data = 'contributions'
            y_scatter_data = 'scatter_y'
            x_kde_data = 'contributions'
            y_kde_data = None
            ax_line = plt.axhline
            ax_line_kwargs = {'y': 0}

        # put into a pandas dataframe and remove minor contribution areas
        df = pd.DataFrame(dict(
            contributions=self.elec_contribs,
            areas=self.anatomy_labels,
            scatter_y=0.0*np.ones((len(self.elec_contribs))),
        ))
        for area in minor_contribution_areas:
            df = df[df['areas'] != area]

        # initialize the FacetGrid object
        rotation = 0.0
        start, fraction, saturation = cubehelix2params(self.RGB_color, rotation)
        self.vprint('CANONICAL COLOR AT FRACTION {0}'.format(fraction))
        palette = sns.cubehelix_palette(
            10, start=start, rot=rotation, light=.7, hue=saturation
        )
        facet_grid = sns.FacetGrid(
            df, hue='areas', palette=palette,
            hue_order=self.anatomy_grand_list, **facet_kwargs
        )

        # draw the densities in a few steps
        facet_grid.map_dataframe(
            getattr(sns, plot_type), x=x_kde_data, y=y_kde_data,
            clip=[0, 1.0], fill=True, alpha=1, lw=1.5, bw_adjust=bw_adjust
        )
        facet_grid.map_dataframe(
            getattr(sns, plot_type), x=x_kde_data, y=y_kde_data,
            clip=[0, 1.0], color="w", lw=2, bw_adjust=bw_adjust
        )
        facet_grid.map(ax_line, **ax_line_kwargs, lw=2, clip_on=False)
        facet_grid.map_dataframe(
            plt.scatter, x=x_scatter_data, y=y_scatter_data, color='black',
            edgecolors='white', linewidths=1.0, s=200.0, zorder=3
        )

        # define function to set the upper bound for all plots
        def set_y_upper_bound(x, lower, label, upper, color):
            ax = plt.gca()
            ax.set_ybound(lower=lower, upper=upper)

        def set_x_upper_bound(x, lower, label, upper, color):
            ax = plt.gca()
            ax.set_xbound(lower=lower, upper=upper)

        # define function to label the plot in axes coordinates
        def label(x, color, label):
            ax = plt.gca()
            this_color = color if label_color is None else label_color
            ax.text(
                1.0, .1, label, fontweight="bold", color=this_color,
                ha="right", va="center", transform=ax.transAxes, fontsize=16
            )

        if VERTICAL:
            facet_grid.map(
                set_x_upper_bound, 'contributions', lower=0, upper=y_upper_bound
            )
            facet_grid.fig.subplots_adjust(wspace=-0.75)
        else:
            facet_grid.map(
                set_y_upper_bound, 'contributions', lower=0, upper=y_upper_bound
            )
            facet_grid.map(label, "contributions")
            facet_grid.fig.subplots_adjust(hspace=-0.75)
            ax = plt.gca()
            ax.set_xlabel('contributions (a.u.)', fontsize=16)

        # remove axes details that don't play well with overlap
        facet_grid.set_titles("")
        facet_grid.set(yticks=[])
        facet_grid.set(xticks=[])
        facet_grid.despine(bottom=True, left=True)

        facet_grid.savefig(self.png_partial_path.format(figure_name), dpi=400)

    def plot_electrode_contributions(
        self, SHOW_X_TICK_LABELS=True, plot_type='boxplot'
    ):
        ##############
        # Is it possible to fold kernel_density_plot_electrode_contributions
        #  and bar_plot_electrode_contributions into this?
        ##############

        # init
        if plot_type == 'violinplot':
            kwargs = {'inner': None, 'bw': 1.0, 'width': 1.0}
        elif plot_type == 'boxplot':
            kwargs = {'showfliers': False}

        figure_name = '_'.join([anatomical_contributions, plot_type])
        minor_contribution_areas = []
        sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

        # put into a pandas dataframe and remove minor contribution areas
        df = pd.DataFrame(dict(
            contributions=self.elec_contribs,
            areas=self.anatomy_labels,
        ))
        for area in minor_contribution_areas:
            df = df[df['areas'] != area]

        # initialize the FacetGrid object
        rotation = 0.0
        start, fraction, saturation = cubehelix2params(self.RGB_color, 0.0)
        self.vprint('CANONICAL COLOR AT FRACTION {0}'.format(fraction))
        palette = sns.cubehelix_palette(
            10, start=start, rot=rotation, light=.7, hue=saturation)

        # ...
        fig, ax = plt.subplots(figsize=(8, 3))
        ax = getattr(sns, plot_type)(
            x='areas', y='contributions', data=df, palette=palette,
            order=self.anatomy_grand_list, ax=ax, **kwargs)
        sns.stripplot(
            x='areas', y='contributions', data=df, order=self.anatomy_grand_list,
            jitter=True, marker='o', s=10, alpha=0.4, color='black')

        ax.set_ybound(lower=0.0, upper=1.1)
        for art in ax.get_children():
            if isinstance(art, mpl.collections.PolyCollection):
                art.set_edgecolor((1.0, 1.0, 1.0))

        ax.set_xticklabels(['{' + item.get_text().replace(' ', '\\\\') + '}'
                            for item in ax.get_xticklabels()])
        ax.set_xlabel(None)

        # export the figure to tikz
        extra_axis_parameters = {
            'width=\\figwidth',
            'height=\\figheight',
            'every x tick label/.append style={rotate=90}',
            'xticklabel style={opacity=\\thisXticklabelopacity, align=center}',
        }
        if not SHOW_X_TICK_LABELS:
            extra_axis_parameters |= {'xticklabels={}'}
        tpl_save(
            filepath=self.tikz_partial_path.format(figure_name),
            extra_axis_parameters=extra_axis_parameters,
            extra_lines_start={
                '\\providecommand{\\figwidth}{5.7in}%',
                '\\providecommand{\\figheight}{2.0in}%',
                '\\providecommand{\\thisXticklabelopacity}{1.0}%',
            },
        )

    ######################
    # BROKEN
    def bar_plot_ecog_sequence_lengths(self, threshold=0.55):

        # save as:
        file_name = 'ecog_sequence_lengths'

        # create a sequence counter
        ######################
        sequence_counters = self.trainer.ecog_subjects[
            -1].get_unique_target_lengths(threshold)
        ######################

        # classify based on length alone, and compute resulting WER
        best_matches = np.argmin(abs(
            np.array(sequence_counters['validation'].lengths_means, ndmin=2) -
            np.array(sequence_counters['training'].lengths_means, ndmin=2).T
        ), axis=0)
        WER = np.mean(wer_vector(
            sequence_counters['training'].unique_sequence_list,
            [sequence_counters['training'].unique_sequence_list[index] for index in best_matches]
        ))

        # plot
        fig, ax = plt.subplots(figsize=(8, 3))
        x_pos = range(len(sequence_counters['training'].lengths_means))
        width = 0.7
        means_and_std_errs = sorted(
            (value, sequence_counters['training'].lengths_std_errs[index])
            for index, value in enumerate(sequence_counters['training'].lengths_means)
        )
        ax.bar(
            x_pos,
            [mean for (mean, std_err) in means_and_std_errs],
            yerr=[std_err for (mean, std_err) in means_and_std_errs],
            width=width,
            color=self.RGB_color
        )
        ax.set_xlabel('sentence ID')
        ax.set_ylabel('length (samples)')
        # ax.set_title('length-based WER: %.1f' % WER*100)

        # save tikzpicture
        tpl_save(filepath=self.tikz_partial_path.format(file_name))

        return WER
    ######################

    def plot_schematic_figures(self):
        '''
        Plot example ECoG, convolutions, hidden encoder states, and predicted
        MFCCs
        '''

        # ...
        assessments = self.get_internal_activations()

        # shortcut notation
        N = self.subject.decimation_factor

        # ...
        iSequence = 0  # why not
        data_list = [
            assessments['validation'].reversed_inputs,
            assessments['validation'].convolved_inputs,
        ]
        signal_names = ['ECoG', 'conv_layer']
        t0s = [N*6, N*10]
        tFs = [N*16, N*11]
        decimation_factors = [1, N]
        nums_channels = [20, 10]
        e0s = [150, 20]
        signal_cmap_names = ['Reds', 'Purples']
        offsets = [0.5, 1.0]
        e_offsets = [None, 0]

        # ...
        filter_color_name = 'green'
        filter_cmap_name = 'Greens'

        def signal_plotter(
            data, cmap_name, t0, tF, num_channels, decimation_factor, e0,
            e_offset, yticks, linewidth=3.0
        ):

            cmap = plt.get_cmap(cmap_name)
            new_prop_cycle = cycler(
                'color', [cmap(i) for i in np.linspace(0.5, 1, num_channels)])
            ax.set_prop_cycle(new_prop_cycle)
            t00 = t0//decimation_factor
            tFF = tF//decimation_factor + 1
            ax.plot(
                range(t00, tFF),
                data[t00:tFF, e0+e_offset:e0+e_offset+num_channels] + yticks,
                linewidth=linewidth
            )

        for (data, signal_name, t0, tF, num_channels, e0, signal_cmap_name, offset,
             e_offset, decimation_factor) in zip(
                data_list, signal_names, t0s, tFs, nums_channels, e0s,
                signal_cmap_names, offsets, e_offsets, decimation_factors):

            # starting the second time thru
            if signal_name != signal_names[0]:
                # overlay filter inputs from previous signals on the previous
                # figure, and save
                signal_plotter(
                    data_prev[iSequence], filter_cmap_name, t0, tF, num_channels_prev,
                    decimation_factor_prev, e0_prev, 0, yticks
                )
                ylim = ax.get_ylim()
                ylen = np.diff(ylim)[0]
                rect = mpl.patches.Rectangle(
                    (t0, ylim[0]), tF-t0, ylen, linewidth=4,
                    edgecolor='green', facecolor='none', label='_nolegend_'
                )
                ax.add_patch(rect)
                tpl_save(filepath=self.tikz_partial_path.format(fig_name))

                # plot with square rather than rectangle
                rect.remove()
                rect = mpl.patches.Rectangle(
                    (t0, ylim[0]+0.2*ylen), tF-t0, ylim[0]+0.3*ylen, linewidth=4,
                    edgecolor='green', facecolor='none', label='_nolegend_'
                )
                ax.add_patch(rect)
                tpl_save(filepath=self.tikz_partial_path.format(fig_name + '_with_square'))

            # now set up the *current* figure
            fig = plt.figure()
            ax = fig.get_axes()[0] if fig.get_axes() else fig.add_subplot(111)
            ax.set_axis_off()

            # plot all signals
            yticks = offset*np.arange(num_channels)
            signal_plotter(
                data[iSequence], signal_cmap_name, t0s[0], tFs[0], num_channels,
                decimation_factor, e0, 0, yticks)

            # starting the second time thru...
            if signal_name != signal_names[0]:
                # overlay (filter) outputs from the previous layer on this figure
                t00 = t0//decimation_factor
                rect = mpl.patches.Rectangle(
                    (t00 - 0.5, data[iSequence][t00, e0+e_offset] + yticks[e_offset] - 1.5),
                    (tF-t0)//decimation_factor, 3.0,
                    linewidth=4, edgecolor='green', facecolor='none',
                    label='_nolegend_'
                )
                ax.add_patch(rect)

                '''
                ax.scatter(
                    t0//decimation_factor,
                    data[iSequence][t0//decimation_factor, e0+e_offset] + yticks[e_offset],
                    facecolor='none', edgecolor=filter_color_name, linewidths=4,
                    s=400, marker='s', label='_nolegend_'
                )
                '''

            # for next time through
            e0_prev = e0
            num_channels_prev = num_channels
            decimation_factor_prev = decimation_factor
            data_prev = data
            fig_name = signal_name + '_example'

            if signal_name == signal_names[0]:
                tpl_save(filepath=self.tikz_partial_path.format(fig_name + '_without_window'))

        # save the last figure
        tpl_save(filepath=self.tikz_partial_path.format(fig_name))

        # also plot encoder final state
        iFinal = 40
        fig = plt.figure(figsize=(5, 2))
        ax = fig.get_axes()[0] if fig.get_axes() else fig.add_subplot(111)
        data = assessments['validation'].final_RNN_state[:, -1, iSequence, :iFinal]
        ax.bar(np.arange(data.shape[1]), np.squeeze(data), color='goldenrod')
        ax.set_axis_off()
        fig_name = 'hidden_state_example'
        tpl_save(filepath=self.tikz_partial_path.format(fig_name))

        # also plot (some) predicted MFCCs
        fig = plt.figure()
        ax = fig.get_axes()[0] if fig.get_axes() else fig.add_subplot(111)
        ax.set_axis_off()
        signal_plotter(
            assessments['validation'].decimated_reversed_targets[iSequence],
            'Blues', t0s[0], tFs[0],
            assessments['validation'].decimated_reversed_targets.shape[2],
            decimation_factor=decimation_factor, e0=0, e_offset=0, yticks=0
        )
        fig_name = 'predicted_MFCCs_example'
        tpl_save(filepath=self.tikz_partial_path.format(fig_name))


    ############
    # BROKEN; PLEASE FIX.  Integrate with all the counters that you have
    # now built!!!    
    def bar_plot_nums_examples(
        self, datasets_to_plot=['training', 'validation'], ymax=None
    ):

        ##########
        # HARD-CODED
        if self.experiment == 'mocha-*':
            num_sentences_dict = {
                1: 50,
                2: 50,
                3: 50,
                4: 50,
                5: 50,
                6: 50,
                7: 50,
                8: 50,
                9: 60
            }
        elif experiment == 'demo2':
            num_sentences_dict = {0: 30}
        else:
            raise NotImplementedError('Oops, haven''t done this yet -- jgm')
        ##########

        # construct usefuls path (see below)
        tf_record_partial_path = self.subject.tf_record_partial_path
      
        # count the number of examples in each MOCHA subset
        subset_counters = {
            k: Counter() for k in self.subject.block_ids.keys()}

        # shorthand
        block_dict = self.subject._block_dict

        ################
        # REPLACE WITH COUNTERS IN subjects.py
        # count up...
        for data_partition in subject.block_ids.keys():
            for block_id in subject.block_ids[data_partition]:
                subkey = (int(block_dict['blocks'][block_id]['type'].split('-')[-1])
                          if experiment == 'mocha-*' else 0)
                subset_counters[data_partition][subkey] += sum(
                    1 for _ in tf_record_iterator(
                        tf_record_partial_path.format(block_id)))

        # now plot
        plt.figure()  # figsize=(2, 5))
        width = 0.7
        total_examples = np.zeros(2*(len(datasets_to_plot)))
        x_inds = np.arange(2*len(datasets_to_plot))
        for subset_id in subset_counters['training'].keys():
            nums_nonunique_examples = [subset_counters[dataset][subset_id]
                                       for dataset in datasets_to_plot]
            nums_unique_examples = [num_sentences_dict[subset_id]
                                    if subset_counters[dataset][subset_id]
                                    else 0 for dataset in datasets_to_plot]
            nums_examples = np.array(nums_nonunique_examples + nums_unique_examples)

            plt.bar(x_inds, nums_examples, bottom=total_examples, width=width,
                    color=elec_color_scheme.hex_colors[subset_id - 1])
            total_examples += nums_examples
        ################


        plt.xticks(x_inds, [dataset + '\n(all)' for dataset in datasets_to_plot] +
                   [dataset + '\n(unique)' for dataset in datasets_to_plot])
        plt.ylabel('\# sentences')
        if ymax is not None:
            axes = plt.gca()
            ylim = axes.get_ylim()
            ylim = axes.set_ylim([ylim[0], ymax])
        #####
        # include total number of words? in training and in validation??
        #####

        ####
        # Write out as tikz file
        ####
        extra_axis_parameters = {
            'xticklabel style={align=center, text width=50}',
            'every x tick label/.append style={rotate=90}',
            'every axis y label/.append style={opacity=\\thisYlabelopacity}'
        }
        extra_lines_start = {'\\providecommand{\\thisYlabelopacity}{1.0}'}
        tpl_save(
            filepath=self.tikz_partial_path.format('data_distribution'),
            extra_axis_parameters=extra_axis_parameters,
            extra_lines_start=extra_lines_start
        )
    ############

    '''
    Methods typically filled in by the MultiSubjectTrainer that instantiates
    this ResultsPlotter object.
    '''
    def get_saliencies():
        return None

    def get_encoder_embedding(self):
        return None


def suffix_to_label(suffix, BOLD_LABEL=False):
    '''
    Translate between naming conventions--one for files (suffix) and the other
    for labels (so e.g. may be space-separated words).
    '''

    id_bits = suffix.split('_')
    label = ''
    '''
    label = id_bits[0]  # the subject of interest
    if 'via' in id_bits:
        transfer_ids = id_bits[id_bits.index('via')+1:-1]
        prefix = '[' + ','.join(transfer_ids) + ']'
        label = prefix + '\\rightarrow' + label
    label += '\\text{ (' + id_bits[-1].upper() + ')}'
    label = '$' + label + '$'  # for tikz
    '''
    if 'cross-subject' in id_bits:
        # bit of a hack....
        label = ' '.join(id_bits[1:-2])
    elif 'via' in id_bits:
        i = id_bits.index('with') + 1
        transfer_ids = []
        while id_bits[i].isdigit():
            transfer_ids.append(id_bits[i])
            i += 1
        if 'mochastar' in id_bits:
            label = '+dual TL (%s)' % ', '.join(transfer_ids)
        else:
            label = '+subject TL (%s)' % ', '.join(transfer_ids)
    elif 'with' in id_bits:
        i = id_bits.index('with') + 1
        transfer_ids = []
        while id_bits[i].isdigit():
            transfer_ids.append(id_bits[i])
            i += 1
        if 'mochastar' in id_bits:
            label = '+dual PTL (%s)' % ', '.join(transfer_ids)
        else:
            label = '+subject PTL (%s)' % ', '.join(transfer_ids)
    elif 'decimated' in id_bits:
        label = 'decimated'
    elif 'untargeted' in id_bits:
        label = 'no MFCCs'
    elif 'undecimated' in id_bits:
        label = 'no conv.'
    elif 'lowdensity' in id_bits:
        label = 'low density'
    elif 'noise' in id_bits:
        label = 'length info. only'
    elif 'mochastar' in id_bits:
        label = '+task TL'
    elif 'viterbi' in id_bits:
        label = 'phoneme-based HMM'
    elif 'cross-attention' in id_bits:
        label = 'cross-attention'
    else:
        label = 'encoder-decoder'

    if BOLD_LABEL:
        label = '{{\\bfseries {0}}}'.format(label)

    return label


class ResultsMatrix:
    @auto_attribute
    def __init__(
        self,
        data
    ):
        pass

    @property
    def mean(self):
        return np.nanmean(self.data, axis=0)

    @property
    def std_err(self):
        return (np.nanvar(self.data, axis=0, ddof=1)/self.data.shape[0])**(1/2)


def plot_performances_vs_amount_of_training_data(
    plotters_list, performance_measures=['word_error_rate', 'accuracy'],
    x_major_ticks=np.arange(0, 41, 10), y_major_ticks=np.arange(6)/5,
    ymin=0.0, ymax=100.0, extra_axis_parameters=None,
    extra_lines_start=None, extra_body_parameters=None,
    fig_num=0, versus='minutes', title=None, line_style_dict=None,
    line_color_legend_loc=1, line_style_legend_loc=9, INCLUDE_LEGEND=True,
    file_infix=None,
):

    # save as:
    file_suffix = 'vs_{0}'.format(versus.replace(' ', '_'))
    png_partial_path = get_save_path('png', plotters_list)
    tikz_partial_path = get_save_path('tikz', plotters_list)

    # for each subject
    for plotter in plotters_list:
        for iPlot, performance_measure in enumerate(performance_measures):
            plotter.line_plot_performance_vs_amount_of_training_data(
                performance_measure, versus, fig_num+iPlot,
                x_major_ticks=x_major_ticks, y_major_ticks=y_major_ticks,
                ymin=ymin, ymax=ymax
            )

    # prepare to output tikz code
    if extra_axis_parameters is None:
        extra_axis_parameters = set()
    extra_axis_parameters = {
        'width=\\figwidth',
        'height=\\figheight',
    } | extra_axis_parameters
    if extra_lines_start is None:
        extra_lines_start = set()
    extra_lines_start = {
        '\\providecommand{\\figwidth}{360pt}',
        '\\providecommand{\\figheight}{310pt}',
    } | extra_lines_start

    # recreate tikzplotlib color names
    tikz_color_names = {}

    # and then put them into a legend via the extra_body_parameters
    if extra_body_parameters is None:
        extra_body_parameters = []

    for i, plotter in enumerate(plotters_list):
        # don't put it in twice, overwriting the old one
        if plotter.subject.subnet_id not in tikz_color_names:
            tikz_color_names[plotter.subject.subnet_id] = 'color{0}'.format(i)
            extra_body_parameters.append(
                '\\addlegendimage{{{0}}}\\addlegendentry{{{1}}}'.format(
                    tikz_color_names[plotter.subject.subnet_id], plotter.alias
                )
            )

    # legend entries for line styles--IN ORDER OF line_style_dict
    if line_style_dict:
        extra_body_parameters.extend(
            ['\\addlegendimage{empty legend}\\addlegendentry{ }'] +
            ['\\addlegendimage{{black, {0}}}\\addlegendentry{{{1}}}'.format(
                line_style, line_style_label)
             for line_style_label, line_style in line_style_dict.items()]
        )

    # SAVE the figures [[Could be done in two loops...]]
    for iPlot, performance_measure in enumerate(performance_measures):
        plt.figure(fig_num+iPlot)
        plt.tight_layout()
        plt.title(title)

        # set up the legend for .png/ipython version
        # NB!!! This also forces the legend for the tikzplotlib version into
        #  the right location.  And you need to force it this way rather than
        #  by setting the axis option "legend pos=north east" b/c adding a
        #  legend also makes tikzplotlib add "forget plot" to all the plots
        #  (which otherwise would get added to the legend).  The *second*
        #  legend is written *below* tpl_save, because it puts a legend in the
        #  *upper middle* of the figure.
        axes = plt.gca()

        if INCLUDE_LEGEND:
            subject_dummy_line_dict = {
                plotter.alias: axes.plot(
                    [], [], c=plotter.RGB_color, ls='solid'
                )[0] for plotter in plotters_list
            }
            subject_legend = plt.legend(
                subject_dummy_line_dict.values(),
                subject_dummy_line_dict.keys(), loc=line_color_legend_loc)
            axes.add_artist(subject_legend)

        # save
        file_name = '_'.join(filter(
            lambda s: s is not None,
            [performance_measure, file_infix, file_suffix]
        ))
        tpl_save(
            filepath=tikz_partial_path.format(file_name),
            extra_axis_parameters=extra_axis_parameters,
            extra_lines_start=extra_lines_start,
            extra_body_parameters=extra_body_parameters,
            strict=True,
        )

        # save
        if line_style_dict:
            line_style_dummy_dict = {
                line_style_label: axes.plot([], [], c="black", ls=line_style)[0]
                for line_style_label, line_style in line_style_dict.items()
            }
            line_style_legend = plt.legend(
                line_style_dummy_dict.values(),
                line_style_dummy_dict.keys(), loc=line_style_legend_loc)
            axes.add_artist(line_style_legend)

        plt.savefig(
            png_partial_path.format(file_name),
            bbox_inches='tight', dpi=400
        )
        ##########
        # patch_thing = plt.gcf().get_children()[0]
        # pdb.set_trace()
        ##########

    return plt


def plot_annotated_performances(
    plotters_list, performance_measures=['word_error_rate', 'accuracy'],
    plot_types=['boxplot'], fig_num=0, file_suffix=None, y_major_ticks=None,
    ymin=0.0, ymax=100.0, title=None, comparisons=None, BOLD_FIRST_LABEL=False,
    VERBOSE=True
):

    # save as:
    if file_suffix is None:
        file_suffix = 'all_data'
    tikz_partial_path = get_save_path('tikz', plotters_list)

    # for writing to tikz
    extra_axis_parameters = {
        'xticklabel style={align=center, text width=50}',
        'every x tick label/.append style={rotate=\\xticklabelangle}',
        'yticklabel style={/pgf/number format/fixed,/pgf/number format/precision=2}',
        'xticklabel style={align=right}',
        'width=\\figwidth',
        'height=\\figheight'
    }
    extra_lines_start = {
        '\\providecommand{\\figwidth}{360pt}',
        '\\providecommand{\\figheight}{310pt}',
        '\\providecommand{\\xticklabelangle}{90}',
        '\\linespread{0}',
    }

    # for each subject
    for iPlot, performance_measure in enumerate(performance_measures):
        for jPlot, plot_type in enumerate(plot_types):
            kPlot = iPlot*len(performance_measures) + jPlot
            ax = plot_performance(
                plotters_list, performance_measure, plot_type, fig_num+kPlot,
                y_major_ticks=y_major_ticks, ymin=ymin, ymax=ymax,
                BOLD_FIRST_LABEL=BOLD_FIRST_LABEL
            )

            if comparisons is not None:

                def application_fxn(
                    baseline_results, comparison_results,
                    baseline_suffix, comparison_suffix,
                ):
                    return pvalue_annotate(
                        baseline_results, comparison_results,
                        baseline_suffix, comparison_suffix,
                        comparisons, ax, plotters_list, ymax, ymin
                    )

                ##################
                # This is kind of a hack:  Here you assume that all results have
                #  the same subnet_id and saved_results_dir.  It's not as bad as
                #  it looks: this is actually already built into apply_comparisons,
                #  and the json files you build to hold your comparisons.  But
                #  ideally you would be able to make comparisons *across* these
                #  variables.
                # NB that you can *plot* across these vars, just not in conjunction
                #  with comparisons not set to None.
                saved_results_partial_path = os.path.join(
                    plotters_list[0].saved_results_dir,
                    'perf_vs_training_size_{0}_{1}.hkl'
                ).format(plotters_list[0].subject.subnet_id, '%s')
                ###############
                apply_comparisons(
                    saved_results_partial_path,
                    comparisons,
                    application_fxn,
                    VERBOSE,
                )

            # plot
            plt.figure(fig_num+kPlot)
            plt.title(title)

            # write to tikz
            # plt.tight_layout()
            tpl_save(
                filepath=tikz_partial_path.format(
                    '_'.join([performance_measure, file_suffix, plot_type])),
                extra_axis_parameters=extra_axis_parameters,
                extra_lines_start=extra_lines_start,
                # strict=True,
            )

    return ax


def plot_performance(
    plotters_list, performance_measure, plot_type, fig_num=0,
    y_major_ticks=None, ymin=0.0, ymax=100.0, BOLD_FIRST_LABEL=False, ax=None
):

    ######
    # TO DO:
    #   (1) Use y_major_ticks
    #   (2) get rid of *100 for other performance measures....
    ######

    # remember to label it as percent (note multiplication by 100 below)
    measure_name = performance_measure.replace('_', ' ') + ' (\%)'

    plt.figure(fig_num)
    df = pd.DataFrame({
        'label': [
            suffix_to_label(
                plotter.suffix,
                (plotter == plotters_list[0]) & BOLD_FIRST_LABEL
            )
            for plotter in plotters_list
            for WER in getattr(plotter, performance_measure).data
        ],
        # convert to percent!
        # NB: last [-1] element of row is from training under all data
        measure_name: [
            WER[-1]*100 for plotter in plotters_list
            for WER in getattr(plotter, performance_measure).data
        ]
    })
    palette = {
        suffix_to_label(
            plotter.suffix,
            (plotter == plotters_list[0]) & BOLD_FIRST_LABEL
        ): plotter.RGB_color
        for plotter in plotters_list
    }

    kwargs = {'ax': ax}
    if plot_type in ['violinplot', 'barplot', 'boxplot', 'swarmplot']:
        kwargs.update({'palette': palette})
    if plot_type == 'violinplot':
        kwargs.update({'inner': 'stick'})
    # elif plot_type == 'swarmplot':
    #     kwargs.update({'color': ".25"})
    #     # kwargs.update({'palette': "dark:.25"})

    ax = getattr(sns, plot_type)(
        x='label', y=measure_name, data=df, hue='label',
        order=[
            suffix_to_label(
                plotter.suffix,
                (plotter == plotters_list[0]) & BOLD_FIRST_LABEL
            )
            for plotter in plotters_list
        ],
        **kwargs
    )
    ax.set(xlabel='')
    ax.set_ylim([ymin, ymax])

    return ax

# for p-value annotation
def pvalue_annotate(
    baseline_results, comparison_results, baseline_suffix, comparison_suffix,
    comparisons, ax, plotters_list, ymax, ymin
):

    # annotation properties
    barh = 0.035
    dh = 0.025
    ceiling = (1.0 - barh - dh)*(ymax - ymin)
    floor = (barh + dh)*(ymax - ymin)

    # [p.get_height() for p in ax.patches],
    heights = (
        [c.get_paths()[0].vertices[1, 1] for c in ax.collections]
        if ax.collections else [0]*len(plotters_list)
    )
    depths = [np.inf]*len(plotters_list)

    for line_obj in ax.lines:
        for x, y in zip(*line_obj.get_data()):
            # some lines are off-center
            x = int(round(x))
            heights[x] = max(heights[x], y)
            try:
                depths[x] = min(depths[x], y)
            except:
                print('This shouldn''t happen')
                pass # pdb.set_trace()

    extrema = heights
    ABOVE = True
    if any([height > ceiling for height in heights]):
        if all([depth > floor for depth in depths]):
            extrema = depths
            ABOVE = False

    # get the locations *in the plot* of the results to be compared
    x0 = [iPlotter for iPlotter, plotter in enumerate(plotters_list)
          if plotter.suffix == baseline_suffix][0]
    x1 = [iPlotter for iPlotter, plotter in enumerate(plotters_list)
          if plotter.suffix == comparison_suffix][0]

    barplot_annotate_brackets(
        x0, x1, comparisons[comparison_suffix][baseline_suffix][
            'adjusted p value'],
        range(len(plotters_list)),
        extrema,
        ABOVE=ABOVE,
        barh=barh,
        dh=dh,
        # dh=annotation_offsets[key],
        max_asterisk=3
    )


def get_save_path(fig_type, plotters_list):

    # get the common path prefix
    common_path = os.path.commonpath([
        os.path.dirname(getattr(plotter, '{}_partial_path'.format(fig_type)))
        for plotter in plotters_list
    ])

    return os.path.join(common_path, os.path.basename(
        getattr(plotters_list[0], '{}_partial_path'.format(fig_type))
    ))


def results_summarizer(summary_path, saved_results_partial_path, VERBOSE=True):
    '''
    Why put this in one function?  So that you can correct for multiple
    comparisons.  The function corrects the p values according to the
    Holm(-Bonferroni) method.
    '''

    # init
    with open(summary_path) as f:
        summary_dict = json.load(f, object_hook=str2int_hook)
    p_values = []

    # Two passes: once to compute stats, then once after Holm-Bonferroni
    #  correction to store the 'adjusted p values'.
    for i in range(2):
        for subj_id, summary_entry in summary_dict.items():
            partial_path = saved_results_partial_path.format(subj_id, '%s')
            for experiment_name, experiment_dict in summary_entry.items():
                if i == 0:
                    # NB that this updates the experiment_dict['comparisons']
                    #  and the p_values (b/c they are passed by reference)
                    results_comparator(
                        partial_path,
                        experiment_dict['comparisons'],
                        p_values,
                        experiment_dict['statistical_test'],
                        VERBOSE,
                    )
                else:
                    # second time through: add in the *adjusted* p values
                    results_updater(
                        partial_path,
                        experiment_dict['comparisons'],
                        ranks,
                        VERBOSE,
                    )
        if i == 0:
            # sort descending and then flip to ascending sort
            ranks = sorted(range(len(p_values)), key=lambda k: p_values[k])
            ranks = [len(ranks) - rank for rank in ranks]

    with open(summary_path, 'w') as f:
        json.dump(summary_dict, f, indent=4)

    return summary_dict


def results_comparator(
    partial_path, comparisons, p_values_list, statistical_test, VERBOSE=True
):

    # this "default statistical test" fxn just throws an error
    def unknown_test(
        word_error_rate_a, word_error_rate_b,
        validation_blocks_a, validation_blocks_b,
    ):
        raise ValueError("Unexpected statistical_test -- jgm")

    # dictionary of possible statistical tests
    tester = {
        'paired t-test': WER_paired_t_test,
        'bootstrap': WER_bootstrap,
        'wilcoxon': WER_wilcoxon,
    }

    def application_fxn(
        baseline_results, comparison_results, baseline_suffix, comparison_suffix
    ):
        # update the dictionary with all stats
        comparisons[comparison_suffix][baseline_suffix] = tester.get(
            statistical_test, unknown_test)(
                baseline_results.word_error_rate,
                comparison_results.word_error_rate,
                baseline_results.validation_blocks,
                comparison_results.validation_blocks
            )

        # Update the list of p_values.  NB that the order of updating is fixed
        #  by apply_comparisons
        p_values_list.append(
            comparisons[comparison_suffix][baseline_suffix]['p value']
        )

    apply_comparisons(partial_path, comparisons, application_fxn, VERBOSE)


def WER_wilcoxon(
    word_error_rate_a, word_error_rate_b,
    validation_blocks_a, validation_blocks_b,
):
    WERs_a = word_error_rate_a.data[:, -1].tolist()
    WERs_b = word_error_rate_b.data[:, -1].tolist()
    blocks_a = validation_blocks_a[:, -1].tolist()
    blocks_b = validation_blocks_b[:, -1].tolist()

    WER_diffs = []
    for blks, wer_a in zip(blocks_a, WERs_a):
        if blks in blocks_b:
            iBlk = blocks_b.index(blks)
            blocks_b.pop(iBlk)
            wer_b = WERs_b.pop(iBlk)
            WER_diffs.append(wer_a - wer_b)
    test_statistic, p = wilcoxon(WER_diffs, alternative='greater')
    # rank-biserial correlation: (T[+] - T[-])/sum(non-zero ranks)
    #                            = (T[+] - T[-])/(T[+] + T[-])
    #                            = 2*T[+]/sum(non-zero ranks) - 1
    effect_size = 2*test_statistic/np.sum(
        np.arange(1, np.sum(np.array(WER_diffs) != 0) + 1)) - 1

    return {
        'p value': p,
        'test statistic': test_statistic,
        'effect size': effect_size
    }


def WER_paired_t_test(
    word_error_rate_a, word_error_rate_b,
    validation_blocks_a, validation_blocks_b,
):
    '''
    t = (xbar1 - xbar2)/sqrt(s1^2/N1 + s2^2/N2)
      = (xbar1 - xbar2)/sqrt(SEM1^2 + SEM2^2)
    nu \approx (s1^2/N1 + s2^2/N2)^2/[s1^4/(N1^2*(N1-1)) + s2^4/(N2^2*(N2-1))]
      =        (SEM1^2 + SEM2^2)^2/[SEM1^4/(N1-1) + SEM2^4/(N2-1)]
    '''

    xbar_a = word_error_rate_a.mean[-1]
    SEM1 = word_error_rate_a.std_err[-1]
    N1 = word_error_rate_a.data.shape[0]

    xbar_b = word_error_rate_b.mean[-1]
    SEM2 = word_error_rate_b.std_err[-1]
    N2 = word_error_rate_b.data.shape[0]

    t = (xbar_b - xbar_a)/(SEM1**2 + SEM2**2)**(1/2)
    nu = (SEM1**2 + SEM2**2)**2/(SEM1**4/(N1-1) + SEM2**4/(N2-1))

    return students_t.cdf(t, nu)


def WER_bootstrap(
    word_error_rate_a, word_error_rate_b,
    validation_blocks_a, validation_blocks_b,
):

    # Ns
    Nrepeats = 1000000
    #####
    # NB: this could actually be wrong, b/c *some* results might use
    #  10-runs per datasize while others use 30!  Add assert statement
    N = validation_blocks_a.shape[0]
    #####

    # for brevity
    blocks_b = validation_blocks_b[:, -1].astype(int)
    blocks_a = validation_blocks_a[:, -1].astype(int)
    unique_blocks = np.unique(blocks_b).tolist()
    unique_blocks == set(np.unique(blocks_a).tolist()) & \
        set(np.unique(blocks_b).tolist())

    # for each unique validation block...
    N_blk_repeats = Nrepeats//len(unique_blocks)
    Nrepeats = N_blk_repeats*len(unique_blocks)
    WER_diff = np.zeros((Nrepeats))
    for i, blk in enumerate(unique_blocks):
        # compute Nrepeats/Nblocks bootstrap estimates...
        WERs_worse = np.random.choice(word_error_rate_a.data[:, -1][
            np.where(blocks_a == blk)[0]], (N_blk_repeats*N))
        WERs_better = np.random.choice(word_error_rate_b.data[:, -1][
            np.where(blocks_b == blk)[0]], (N_blk_repeats*N))
        # ...of the mean WER differences, and store
        WER_diff[i*N_blk_repeats:(i+1)*N_blk_repeats] = np.mean(np.reshape(
            WERs_worse - WERs_better, (N_blk_repeats, N)), axis=1)

    # altogether we have Nblocks of these Nrepeats/Nblocks estimates,
    #  i.e. Nrepeats; return the position of 0 in their empirical dstrb
    return np.mean(WER_diff < 0.0)


def results_updater(partial_path, comparisons, ranks, VERBOSE):

    def application_fxn(
        baseline_results, comparison_results, baseline_suffix, comparison_suffix
    ):
        # Holm-Bonferroni correction
        rank = ranks.pop(0)
        comparisons[comparison_suffix][baseline_suffix]['adjusted p value'] = \
            comparisons[comparison_suffix][baseline_suffix]['p value']*rank

        comparisons[comparison_suffix][baseline_suffix]['rank'] = rank

    apply_comparisons(partial_path, comparisons, application_fxn, VERBOSE)


def apply_comparisons(partial_path, comparisons, application_fxn, VERBOSE):
    '''
    Make all the requested comparisons, applying the application_fxn each time.
    '''

    for comparison_suffix, baseline_dict in comparisons.items():
        for baseline_suffix in baseline_dict.keys():

            # construct DecodingResults objects for comparison and baseline
            comparison_results = DecodingResults(
                partial_path % comparison_suffix, VERBOSE)
            baseline_results = DecodingResults(
                partial_path % baseline_suffix, VERBOSE)

            # ...
            application_fxn(
                baseline_results, comparison_results,
                baseline_suffix, comparison_suffix,
            )


def ith_param_range(grids, ii, grid_shape):
    ith_grid = np.reshape(grids[:, ii], grid_shape)
    return np.reshape(np.moveaxis(ith_grid, ii, -1), [-1])[:ith_grid.shape[ii]]


def project_grid_search(
    marginal_params, conditioning_list, performance, parameter_names, grids,
    grid_shape, ax
):

    # check
    assert np.prod(grid_shape) == grids.shape[0], (
            "grid_shape doesn't match grids.shape[0")

    # init
    vmax = np.nanmax(performance)
    performance = np.reshape(performance, grid_shape)
    nonmarginal_params = []
    heatmap_axes = []

    # loop through parameters (dimensions) of the grid search
    for iParam, parameter in enumerate(parameter_names):
        param_range = ith_param_range(grids, iParam, grid_shape)
        if parameter in marginal_params:
            cond = np.array(conditioning_list)[parameter == np.array(marginal_params)][0]
            if cond is None:
                # print('marginalizing out %s' % parameter)
                performance = np.mean(performance, axis=iParam, keepdims=True)
            else:
                # print('conditioning on %s = %i' % (parameter, cond))
                indexer = [slice(None)]*performance.ndim
                indexer[iParam] = np.where(param_range == cond)[0]
                performance = performance[tuple(indexer)]
        else:
            nonmarginal_params.append(parameter)
            heatmap_axes.append(param_range)

    # seaborn's heatmap
    sns.heatmap(
        np.squeeze(performance), square=True, annot=True, ax=ax,
        cbar=False, vmin=0, vmax=vmax, mask=np.isnan(np.squeeze(performance)),
        xticklabels=heatmap_axes[1], yticklabels=heatmap_axes[0]
    )
    ax.set(xlabel=nonmarginal_params[1], ylabel=nonmarginal_params[0])

    return performance


def all_grid_search_projections(
    grid_shape, marginal_params, subj_id, saved_results_dir, suffix=''
):
    ######
    # Integrate with ResultsPlotter?
    # NB that the saved_results_dir is in the manifest
    ######

    # load from file
    loadfile_name = os.path.join(
        saved_results_dir, 'grid_search_{0}_conv_{1}_way{2}.hkl'.format(
            subj_id, len(grid_shape), suffix)
    )
    all_results, parameter_names, grids = hickle.load(loadfile_name)

    # plot all
    ranges = []
    for marginal_param in marginal_params:
        ii = np.where(np.array(parameter_names) == marginal_param)[0][0]
        ranges.append(ith_param_range(grids, ii, grid_shape))

    # the ranges of the last two marginal_params set the shape of the figure
    subplot_shape = [len(rng) for rng in ranges[-2:]]
    outer_params = marginal_params[:-2]
    inner_params = marginal_params[-2:]

    # loop
    for ii, conditioning_tuple in enumerate(itertools.product(*ranges)):
        jj = ii % np.prod(subplot_shape)
        if jj == 0:
            fig = plt.figure(figsize=(20, 20))
            title_bit = ', '.join([
                '{0}={1}'.format(param, cond) for param, cond
                in zip(outer_params, conditioning_tuple[:-2])
            ])
            fig.suptitle('Performance: ' + title_bit, fontsize=16)
            fig.text(
                0.5, 0.04, inner_params[1], ha="center", va="center",
                fontsize=12
            )
            fig.text(
                0.05, 0.5, inner_params[0], ha="center", va="center",
                rotation=90, fontsize=12
            )

        ax = fig.add_subplot(*subplot_shape, jj+1)
        ###plt.tight_layout()
        performance = project_grid_search(
            marginal_params, list(conditioning_tuple),
            all_results['word_error_rate'], parameter_names, grids,
            grid_shape, ax)


def print_latex_anatomical_legend():

    from img_pipe.SupplementalFiles import FS_colorLUT as FS_colorLUT

    # hard-coded
    brain_areas = [
        'precentral', 'postcentral', 'supramarginal',   # M1, S1, PPC
        'pars triangularis', 'pars opercularis', 'pars orbitalis',  # IFG
        'superior temporal', 'middle temporal', 'inferior temporal',  # temporal
        'rostral middle frontal', 'caudal middle frontal',  # middle front
    ]
    legend_split_points = ['pars orbitalis']
    label_hem = 'lh'  # colors seem to be the same for both

    # get the colors in RGB
    cmap = FS_colorLUT.get_lut()
    color_list = [
        np.array(cmap['ctx-{0}-{1}'.format(label_hem, area.replace(' ',''))])
        for area in brain_areas
    ]
    RGB_colors = np.stack(color_list)

    # print LaTeX code defining the new colors
    for color, area in zip(RGB_colors, brain_areas):
        print('\providecolor{{{0}}}{{RGB}}{{{1},{2},{3}}}'.format(area.replace(' ',''), *color))
    print('')

    # print LaTeX/TikZ code creating a command->figure->legend
    print('\\newcommand{\\anatomyLegend}{%')
    print('\t\\begin{tikzpicture}[>=latex,remember picture]%')
    vertical_pos = 0

    print('\t\t\\node at (0,{0}) {{%'.format(vertical_pos))
    while brain_areas:
        # this area's legend entry
        area = brain_areas.pop(0)
        print('\t\t\t\\begin{tikzpicture}')
        print('\t\t\t\t\\fill[{0}] (1ex,1ex) circle (1ex)'.format(area.replace(' ','')))
        print('\t\t\t\t\tnode[label={{[black]right:{0}}}] {{}};'.format(area))
        print('\t\t\t\\end{tikzpicture}')

        # start a new node?
        if area in legend_split_points:
            # close off the node and shift the next node down
            print('\t\t};')
            vertical_pos -= 0.5
            print('\t\t\\node at (0,{0}) {{%'.format(vertical_pos))

    print('\t\t};')
    print('\t\\end{tikzpicture}')
    print('}')


########
# This might be (or could be made) general enough to be moved out of here
def cluster_embeddings(
    M, num_reduced_dims=2, num_mixture_components=3, POLAR=False,
    dimensionality_reducer='PCA', num_PCs_for_tSNE=50, data_labels=None,
    fig_dir='', file_name='word_embeddings',
):

    from sklearn.mixture import GaussianMixture
    from sklearn.manifold import TSNE

    # singular-value decomposition
    Mcntrd = M - np.sum(M, axis=0, keepdims=True)/M.shape[0]
    U, s, Vtr = np.linalg.svd(Mcntrd)

    # dimensionality reduction
    if dimensionality_reducer == 'PCA':
        reduced_embedding = M@Vtr[:num_reduced_dims, :].T
        if POLAR and (num_reduced_dims == 2):
            reduced_embedding = np.stack(
                (np.sqrt(np.sum(reduced_embedding**2, axis=1)),
                 np.arctan2(reduced_embedding[:, 1], reduced_embedding[:, 0])),
                axis=1
            )
    elif dimensionality_reducer == 't-SNE':
        reduced_embedding = M@Vtr[:num_PCs_for_tSNE, :].T
        reduced_embedding = TSNE(n_components=num_reduced_dims).fit_transform(
            reduced_embedding)
    else:
        raise ValueError("Unexpected dimensionality_reducer -- jgm")

    # fit a GMM to words in reduced, embedded space, and then classify them
    GMM = GaussianMixture(n_components=num_mixture_components)
    GMM.fit(reduced_embedding)
    class_labels = GMM.predict(reduced_embedding)

    # plot the singular values
    fig1 = plt.figure(figsize=(5, 5))
    plt.plot(s)

    # plot the word embeddings in the reduced space
    # fig2 = plt.figure(figsize=(15, 15))
    # ax = Axes3D(fig2)
    # ax.scatter(*reduced_embedding[:,:3].T, c=class_labels)
    # ax.view_init(60, 80)
    # ax.view_init(90, 90)
    if data_labels is None:
        fig2 = plt.figure(figsize=(15, 15))
        ax = fig2.add_subplot(1, 1, 1)
        ax.scatter(*reduced_embedding.T, c=class_labels)
    else:
        if len(data_labels) != reduced_embedding.shape[0]:
            # assume they are sequence lengths
            fig2 = scatter_desequenced_data(
                reduced_embedding, data_labels, figsize=(15, 15))
        else:
            fig2 = plt.figure(figsize=(15, 15))
            ax = fig2.add_subplot(1, 1, 1)
            ax.scatter(*reduced_embedding.T, c=class_labels)
            for iLabel, label in enumerate(data_labels):
                ax.annotate(
                    label,
                    *reduced_embedding[None, iLabel, :],
                    xycoords="data", va="center", ha="center"
                )
    fig2.savefig(os.path.join(fig_dir, file_name))

    return class_labels, reduced_embedding
########


def scatter_desequenced_data(
    desequenced_data, sequence_lengths, sequence_ids=None, figsize=(5, 5)
):

    # set up figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)

    # ...
    if sequence_ids is None:
        sequence_ids = range(len(sequence_lengths))

    # loop through sequences
    i0 = 0
    for iSequence, length in enumerate(sequence_lengths):
        iF = i0 + length
        if iSequence in sequence_ids:
            this_axis_list = ax.plot(*desequenced_data[i0:iF, :].T)
            color = this_axis_list[0].get_color()
            ax.scatter(*desequenced_data[i0, :].T, marker='o', c=color)
            ax.scatter(*desequenced_data[iF-1, :].T, marker='^', c=color)
            ax.annotate(
                iSequence,
                *desequenced_data[None, i0, :],
                xycoords="data", va="center", ha="center",  # color=color
            )
            ax.annotate(
                iSequence,
                *desequenced_data[None, iF-1, :],
                xycoords="data", va="center", ha="center", fontweight='bold'
            )
        i0 = iF

    return fig


def effect_of_tabularizer(summary_dict):

    # first make sure that all subjects were compared to the same "baselines"
    for iSubj, (subj_id, summary_entry) in enumerate(summary_dict.items()):
        # there should be only one of these:
        for experiment_dict in summary_entry.values():
            # there should be only one of these:
            for comparisons in experiment_dict['comparisons'].values():
                tentative_labels = [suffix_to_label(suffix)
                                    for suffix in comparisons.keys()]
                if iSubj == 0:
                    labels = tentative_labels
                    hline = '\\\\\hline'
                    print('\\begin{tabular}{r%s}' % ('|c'*(len(labels)+1)))
                    print('participant & baseline: & ' + ' & '.join(labels)
                          + hline + '\\hline')
                else:
                    assert labels == tentative_labels

                # adjusted_p_values = ['%.1e' % stats['adjusted p value']
                #                      for stats in comparisons.values()]
                p_values = [
                    '%.1e' % stats['p value']
                    for stats in comparisons.values()
                ]
                test_statistics = [
                    '%d' % stats['test statistic']
                    for stats in comparisons.values()
                ]
                effect_sizes = [
                    '%.2f' % stats['effect size']
                    for stats in comparisons.values()
                ]

                cline = '\\\\\\cline{2-%i}' % (len(labels) + 2)

                print('\\ecnum{%i} & (unadjusted) p value & ' % subj_id
                      + ' & '.join(p_values) + cline)
                print('           & test statistic & '
                      + ' & '.join(test_statistics) + cline)
                print('           & effect size & ' + ' & '.join(effect_sizes)
                      + hline)
    print('\\end{tabular}')
