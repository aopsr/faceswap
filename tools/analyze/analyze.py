#!/usr/bin/env python3
"""
A tool that allows for analyzing and grouping images in different ways.
"""
from __future__ import annotations
import logging
import os
import sys
import typing as T
import numpy as np

from argparse import Namespace
from shutil import copyfile, rmtree

from tqdm import tqdm

# faceswap imports
from lib.serializer import Serializer, get_serializer_from_filename
from lib.utils import deprecation_warning

from lib.align import AlignedFace, DetectedFace, PoseEstimate
from lib.image import FacesLoader, ImagesLoader, read_image_meta_batch, update_existing_metadata
from lib.utils import FaceswapError

logger = logging.getLogger(__name__)


class Analyze():  # pylint:disable=too-few-public-methods
    """ analyzes folders of faces based on input criteria

    Wrapper for the analyze process to run in either batch mode or single use mode

    Parameters
    ----------
    arguments: :class:`argparse.Namespace`
        The arguments to be passed to the extraction process as generated from Faceswap's command
        line arguments
    """
    def __init__(self, arguments: Namespace) -> None:
        logger.debug("Initializing: %s (args: %s)", self.__class__.__name__, arguments)
        self._args = arguments
        self._handle_deprecations()
        self._input_locations = self._get_input_locations()
        logger.debug("Initialized: %s", self.__class__.__name__)

    def _handle_deprecations(self):
        """ Warn that 'final_process' is deprecated and remove from arguments """
        pass

    def _get_input_locations(self) -> list[str]:
        """ Obtain the full path to input locations. Will be a list of locations if batch mode is
        selected, or a containing a single location if batch mode is not selected.

        Returns
        -------
        list:
            The list of input location paths
        """
        if not self._args.batch_mode:
            return [self._args.input_dir]

        retval = [os.path.join(self._args.input_dir, fname)
                  for fname in os.listdir(self._args.input_dir)
                  if os.path.isdir(os.path.join(self._args.input_dir, fname))]
        logger.debug("Input locations: %s", retval)
        return retval

    def process(self) -> None:
        """ The entry point for triggering the analyze Process.

        Should only be called from  :class:`lib.cli.launcher.ScriptExecutor`
        """
        logger.info('Starting, this may take a while...')
        inputs = self._input_locations
        if self._args.batch_mode:
            logger.info("Batch mode selected processing: %s", self._input_locations)
        for job_no, location in enumerate(self._input_locations):
            if self._args.batch_mode:
                logger.info("Processing job %s of %s: '%s'", job_no + 1, len(inputs), location)
                arguments = Namespace(**self._args.__dict__)
                arguments.input_dir = location
            else:
                arguments = self._args
            analyze = _Analyze(arguments)
            analyze.process()


class _Analyze():  # pylint:disable=too-few-public-methods
    """ analyzes folders of faces based on input criteria """
    def __init__(self, arguments: Namespace) -> None:
        logger.debug("Initializing %s: arguments: %s", self.__class__.__name__, arguments)

        self._args = arguments
        self._changes: dict[str, str] = {}
        self.serializer: Serializer | None = None
        
        self._loader = FacesLoader(self._args.input_dir)
        self._cached_source_data: dict[str, PNGHeaderSourceDict] = {}

        logger.debug("Initialized %s", self.__class__.__name__)
    
    def process(self) -> None:
        """ Main processing function of the analyze tool

        """
        metadata = self._metadata_reader()
        
        pitch = []
        yaw = []
        for m in metadata:
            landmarks = m[2]["landmarks_xy"]
            face = AlignedFace(landmarks = np.array(landmarks))
            face.pose._get_pitch_yaw_roll()
            print(face.pose._pitch_yaw_roll)
            pitch.append(face.pose._pitch_yaw_roll[0])
            yaw.append(face.pose._pitch_yaw_roll[1])
        
        import matplotlib.pyplot as plt

        plt.hist2d(pitch, yaw, bins=(50, 50), cmap=plt.cm.jet)
        plt.show()
        
    def _get_alignments(self,
                    filename: str,
                    metadata: dict[str, T.Any]) -> PNGHeaderAlignmentsDict | None:
        """ Obtain the alignments from a PNG Header.

        The other image metadata is cached locally in case a analyze method needs to write back to the
        PNG header

        Parameters
        ----------
        filename: str
            Full path to the image PNG file
        metadata: dict
            The header data from a PNG file

        Returns
        -------
        dict or ``None``
            The alignments dictionary from the PNG header, if it exists, otherwise ``None``
        """
        if not metadata or not metadata.get("alignments") or not metadata.get("source"):
            return None
        self._cached_source_data[filename] = metadata["source"]
        return metadata["alignments"]

    def _metadata_reader(self) -> ImgMetaType:
        """ Load metadata from saved aligned faces

        Yields
        ------
        filename: str
            The filename that has been read
        image: None
            This will always be ``None`` with the metadata reader
        alignments: dict or ``None``
            The alignment data for the given face or ``None`` if no alignments found
        """
        for filename, metadata in tqdm(read_image_meta_batch(self._loader.file_list),
                                        total=self._loader.count,
                                        leave=False):
            alignments = self._get_alignments(filename, metadata.get("itxt", {}))
            yield filename, None, alignments
