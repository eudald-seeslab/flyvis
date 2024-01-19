import os

from typing import List
from tqdm import tqdm
import torch
import numpy as np

import matplotlib.pyplot as plt

import flyvision
from flyvision import EnsembleView, NetworkView
from flyvision.animations import StimulusResponse
from flyvision.datasets.base import SequenceDataset
from flyvision.utils.activity_utils import LayerActivity

np.random.seed(42)

DECODING_CELLS = ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d", "T1", "T2", "T2a", "T3", "Tm1", "Tm2",
                 "Tm3", "Tm4", "Tm5Y", "Tm5a", "Tm5b", "Tm5c", "Tm9", "Tm16", "Tm20", "Tm28", "Tm30", "TmY3", "TmY4",
                 "TmY5a", "TmY9", "TmY10", "TmY13", "TmY14", "TmY15", "TmY18"]
FINAL_CELLS = ["TmY15", "TmY16"]


def load_custom_sequences(video_dir):
    videos = [np.load(os.path.join(video_dir, a)) for a in os.listdir(video_dir)]
    # Take the average of the 3 color channels to get a single greyscale channel
    # Create an array with the proper dimensions (n_sequences, n_frames, height, width)
    return np.array([np.mean(a, axis=3) for a in videos])


class RenderedData:

    def __init__(self, input_path, extent, kernel_size, subset_idx=None):
        # here comes the preprocessing and rendering as above or similar -- depending on the dataset etc.
        # this code will be executed automatically once for each unique configuration to store preprocessed
        # data on disk and later simply provide a reference to it.
        if subset_idx is None:
            subset_idx = []
        sequences = load_custom_sequences(input_path)

        # we use the configuration to control the settings under which we render the stimuli
        receptors = flyvision.rendering.BoxEye(
            extent=extent, kernel_size=kernel_size
        )

        # for memory-friendly rendering we can loop over individual sequences
        # and subsets of the dataset
        rendered_sequences = []
        subset_idx = subset_idx if len(subset_idx) > 0 else list(range(sequences.shape[0]))
        with tqdm(total=len(subset_idx)) as pbar:
            for index in subset_idx:
                rendered_sequences.append(receptors(sequences[[index]]).cpu().numpy())
                pbar.update()

        # to join individual sequences along their first dimension
        # to obtain (n_sequences, n_frames, 1, receptors.hexals)
        rendered_sequences = np.concatenate(rendered_sequences, axis=0)

        # the __setattr__ method of the Directory class saves sequences to self.path/"sequences.h5"
        # that can be later retrieved using self.sequences[:]
        self.sequences = rendered_sequences


class CustomStimuli(SequenceDataset):
    # implementing the SequenceDataset interface
    dt = 1 / 100
    framerate = 24
    t_pre = 0.5
    t_post = 0.5
    n_sequences = None
    augment = False

    def __init__(self, input_path, extent, kernel_size, subset_idx):
        self.dir = RenderedData(input_path, extent, kernel_size, subset_idx)
        self.sequences = torch.Tensor(self.dir.sequences[:])
        self.n_sequences = self.sequences.shape[0]

    def get_item(self, key):
        sequence = self.sequences[key]
        # to match the framerate to the integration time dt, we can resample frames
        # from these indices. note, when dt = 1/framerate, this will return the exact sequence
        resample = self.get_temporal_sample_indices(sequence.shape[0], sequence.shape[0])
        return sequence[resample]


class ResponseProcessor:
    """
    Does the work of processing batches of input images into cell responses
    """

    data = None
    movie_input = None

    def __init__(self, input_data_path):
        self.input_path = input_data_path
        network_view = NetworkView(flyvision.results_dir / "opticflow/000/0000")
        self.network = network_view.init_network(chkpt="best_chkpt")
        self.ensemble = EnsembleView(flyvision.results_dir / "opticflow/000")

    def compute_responses(self):
        data = CustomStimuli(input_path="../videos/yellow", extent=15, kernel_size=13, subset_idx=[])

        # ensemble.simulate returns an iterator over `network.simulate` for each network.
        # we exhaust it and stack responses from all models in the first dimension
        return [self.network.simulate(a[None], data.dt) for a in data]

    def compute_layer_activations(self, _responses=None):
        if _responses is None:
            _responses = self.compute_responses()

        la = []
        for response in _responses:
            la.append(LayerActivity(response, self.network.connectome, keepref=True))
        return la

    def compute_animations(self, cell_type, _responses=None):
        if _responses is None:
            _responses = self.compute_responses()
        return StimulusResponse(
            self.movie_input[None],
            _responses[cell_type][:, :, None]
        )

    def animate_responses(self, cell_type, _responses=None):
        animations = self.compute_animations(cell_type, _responses)
        return animations.animate_in_notebook(frames=np.arange(animations.frames)[::2])


if __name__ == "__main__":
    # load the data
    response_processor = ResponseProcessor("../videos/yellow")
    # compute the responses
    responses = response_processor.compute_responses()
    # compute the layer activations
    layer_activations = response_processor.compute_layer_activations(responses)
    # save layer activations from decoding cells


    # get the animation objects to inspect them
    animations = response_processor.compute_animations("TmY15", responses)
    # animate the responses
    response_processor.animate_responses("TmY15", responses)
