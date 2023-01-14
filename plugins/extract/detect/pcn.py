#!/usr/bin/env python3
""" PCN Face detection plugin """
from __future__ import absolute_import, division, print_function
import logging
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from lib.model.session import KSession
from lib.utils import get_backend
from ._base import BatchType, Detector

if get_backend() == "amd":
    from keras.layers import Conv2D, Dense, Flatten, Input, MaxPool2D, Permute, PReLU, Softmax
    from plaidml.tile import Value as Tensor  # pylint:disable=import-error
else:
    # Ignore linting errors from Tensorflow's thoroughly broken import system
    from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPool2D, Permute, PReLU, ReLU, Softmax  # noqa pylint:disable=no-name-in-module,import-error
    from tensorflow import Tensor
    import tensorflow as tf

logger = logging.getLogger(__name__)

# global settings
EPS = 1e-5
minFace_ = 20 * 1.4
scale_ = 1.414
stride_ = 8
classThreshold_ = [0.37, 0.43, 0.97]
nmsThreshold_ = [0.8, 0.8, 0.3]
angleRange_ = 45
stable_ = 0

class Window:
    def __init__(self, x, y, width, angle, score):
        self.x = x
        self.y = y
        self.width = width
        self.angle = angle
        self.score = score

class Window2:
    def __init__(self, x, y, w, h, angle, scale, conf):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.angle = angle
        self.scale = scale
        self.conf = conf

class Detect(Detector):
    """ PCN detector for face recognition. """
    def __init__(self, **kwargs) -> None:
        git_model_id = 2
        model_filename = ["mtcnn_det_v2.1.h5", "mtcnn_det_v2.2.h5", "mtcnn_det_v2.3.h5"]
        super().__init__(git_model_id=git_model_id, model_filename=model_filename, **kwargs)
        self.name = "PCN"
        self.input_size = 640
        self.vram = 320 if not self.config["cpu"] else 0
        self.vram_warnings = 64 if not self.config["cpu"] else 0  # Will run at this with warnings
        self.vram_per_batch = 32 if not self.config["cpu"] else 0
        self.batchsize = self.config["batch-size"]
        self.kwargs = self._validate_kwargs()
        self.color_format = "RGB"

    def _validate_kwargs(self) -> Dict[str, Union[int, float, List[float]]]:
        """ Validate that config options are correct. If not reset to default """
        valid = True
        threshold = [self.config["threshold_1"],
                     self.config["threshold_2"],
                     self.config["threshold_3"]]
        kwargs = {"minsize": self.config["minsize"],
                  "threshold": threshold,
                  "factor": self.config["scalefactor"],
                  "input_size": self.input_size}

        if kwargs["minsize"] < 10:
            valid = False
        elif not all(0.0 < threshold <= 1.0 for threshold in kwargs['threshold']):
            valid = False
        elif not 0.0 < kwargs['factor'] < 1.0:
            valid = False

        if not valid:
            kwargs = {}
            logger.warning("Invalid MTCNN options in config. Running with defaults")

        logger.debug("Using PCN kwargs: %s", kwargs)
        return kwargs

    def init_model(self) -> None:
        """ Initialize PCN Model. """
        assert isinstance(self.model_path, list)
        self.model = PCN(self.model_path,
                           self.config["allow_growth"],
                           self._exclude_gpus,
                           self.config["cpu"],
                           **self.kwargs)  # type:ignore

    def process_input(self, batch: BatchType) -> None:
        """ Compile the detection image(s) for prediction

        Parameters
        ----------
        batch: :class:`~plugins.extract.detect._base.DetectorBatch`
            Contains the batch that is currently being passed through the plugin process
        """
        batch.feed = (np.array(batch.image, dtype="float32") - 127.5) / 127.5

    def predict(self, feed: np.ndarray) -> np.ndarray:
        """ Run model to get predictions

        Parameters
        ----------
        batch: :class:`~plugins.extract.detect._base.DetectorBatch`
            Contains the batch to pass through the MTCNN model

        Returns
        -------
        dict
            The batch with the predictions added to the dictionary
        """
        assert isinstance(self.model, PCN)
        prediction, points = self.model.detect_faces(feed)
        logger.trace("prediction: %s, mtcnn_points: %s",  # type:ignore
                     prediction, points)
        return prediction

    def process_output(self, batch: BatchType) -> None:
        """ MTCNN performs no post processing so the original batch is returned

        Parameters
        ----------
        batch: :class:`~plugins.extract.detect._base.DetectorBatch`
            Contains the batch to apply postprocessing to
        """
        return

class PCN1(KSession):

    def __init__(self,
                 model_path: str,
                 allow_growth: bool,
                 exclude_gpus: Optional[List[int]],
                 cpu_mode: bool,
                 input_size: int,
                 threshold: float) -> None:
        super().__init__("PCN1",
                         model_path,
                         allow_growth=allow_growth,
                         exclude_gpus=exclude_gpus,
                         cpu_mode=cpu_mode)
        self.define_model(self.model_definition)
        self.load_model_weights()

    @staticmethod
    def model_definition():
        x = Input(shape=(None, None, 3))
        x = Conv2D(16, kernel_size=3, stride=2, activation="relu")(x)
        x = Conv2D(32, kernel_size=3, stride=2, activation="relu")(x)
        x = Conv2D(64, kernel_size=3, stride=2, activation="relu")(x)
        x = Conv2D(128, kernel_size=2, stride=1, activation="relu")(x)
        cls_prob = Softmax(1)(Conv2D(2, kernel_size=1, stride=1)(x))
        rotate = Softmax(1)(Conv2D(2, kernel_size=1, stride=1)(x))
        bbox = Conv2D(3, kernel_size=1, stride=1)(x)
        return cls_prob, rotate, bbox

class PCN2(KSession):

    def __init__(self,
                 model_path: str,
                 allow_growth: bool,
                 exclude_gpus: Optional[List[int]],
                 cpu_mode: bool,
                 input_size: int,
                 threshold: float) -> None:
        super().__init__("PCN2",
                         model_path,
                         allow_growth=allow_growth,
                         exclude_gpus=exclude_gpus,
                         cpu_mode=cpu_mode)
        self.define_model(self.model_definition)
        self.load_model_weights()

    @staticmethod
    def model_definition():
        x = Conv2D(20, kernel_size=3, stride=1)(x)
        x = tf.pad(x, [[0, 1], [0, 1]])
        #x = F.pad(x, (0, 1, 0, 1))
        x = ReLU()(MaxPool2D(kernel_size=3, stride=2)(x))
        x = Conv2D(40, kernel_size=3, stride=1)(x)
        x = tf.pad(x, [[0, 1], [0, 1]])
        #x = F.pad(x, (0, 1, 0, 1))
        x = ReLU()(MaxPool2D(kernel_size=3, stride=2)(x))
        x = Conv2D(70, kernel_size=2, stride=1, activatin="relu")(x)
        #x = x.view(batch_size, -1)
        x = Flatten()(x)
        x = Dense(140, activation="relu")(x)
        cls_prob = Softmax(1)(Dense(2)(x))
        rotate = Softmax(1)(Dense(3)(x))
        bbox = Dense(3)(x)
        return cls_prob, rotate, bbox

class PCN3(KSession):

    def __init__(self,
                 model_path: str,
                 allow_growth: bool,
                 exclude_gpus: Optional[List[int]],
                 cpu_mode: bool,
                 input_size: int,
                 threshold: float) -> None:
        super().__init__("PCN3",
                         model_path,
                         allow_growth=allow_growth,
                         exclude_gpus=exclude_gpus,
                         cpu_mode=cpu_mode)
        self.define_model(self.model_definition)
        self.load_model_weights()


    @staticmethod
    def model_definition():
        x = Conv2D(24, kernel_size=3, stride=1)(x)
        x = tf.pad(x, [[0, 1], [0, 1]])
        x = ReLU()(MaxPool2D(kernel_size=3, stride=2)(x))

        x = Conv2D(48, kernel_size=3, stride=1)(x)
        x = tf.pad(x, [[0, 1], [0, 1]])
        x = ReLU()(MaxPool2D(kernel_size=3, stride=2)(x))

        x = Conv2D(96, kernel_size=3, stride=1)(x)
        x = ReLU()(MaxPool2D(kernel_size=2, stride=2)(x))
        x = Conv2D(144, kernel_size=2, stride=1, activation="relu")(x)
        x = Flatten()(x)
        x = ReLU()(Dense(192)(x))
        cls_prob = Softmax(1)(Dense(2)(x))
        rotate = Dense(1)(x)
        bbox = Dense(3)(x)
        return cls_prob, rotate, bbox

class PCN():  # pylint: disable=too-few-public-methods
    """ PCN Detector for face alignment

    Parameters
    ----------
    model_path: list
        List of paths to the 3 PCN subnet weights
    allow_growth: bool, optional
        Enable the Tensorflow GPU allow_growth configuration option. This option prevents
        Tensorflow from allocating all of the GPU VRAM, but can lead to higher fragmentation and
        slower performance. Default: ``False``
    exclude_gpus: list, optional
        A list of indices correlating to connected GPUs that Tensorflow should not use. Pass
        ``None`` to not exclude any GPUs. Default: ``None``
    cpu_mode: bool, optional
        ``True`` run the model on CPU. Default: ``False``
    input_size: int, optional
        The height, width input size to the model. Default: 640
    minsize: int, optional
        The minimum size of a face to accept as a detection. Default: `20`
    threshold: list, optional
        List of floats for the three steps, Default: `[0.6, 0.7, 0.7]`
    factor: float, optional
        The factor used to create a scaling pyramid of face sizes to detect in the image.
        Default: `0.709`
    """
    def __init__(self,
                 model_path: List[str],
                 allow_growth: bool,
                 exclude_gpus: Optional[List[int]],
                 cpu_mode: bool,
                 input_size: int = 640,
                 minsize: int = 20,
                 threshold: Optional[List[float]] = None,
                 factor: float = 0.709) -> None:
        logger.debug("Initializing: %s: (model_path: '%s', allow_growth: %s, exclude_gpus: %s, "
                     "input_size: %s, minsize: %s, threshold: %s, factor: %s)",
                     self.__class__.__name__, model_path, allow_growth, exclude_gpus,
                     input_size, minsize, threshold, factor)

        threshold = [0.6, 0.7, 0.7] if threshold is None else threshold
        self.pcn1 = PCN1(model_path[0],
                          allow_growth,
                          exclude_gpus,
                          cpu_mode,
                          input_size,
                          minsize,
                          factor,
                          threshold[0])
        self.pcn2 = PCN2(model_path[1],
                          allow_growth,
                          exclude_gpus,
                          cpu_mode,
                          input_size,
                          threshold[1])
        self.pcn3 = PCN3(model_path[2],
                          allow_growth,
                          exclude_gpus,
                          cpu_mode,
                          input_size,
                          threshold[2])

        logger.debug("Initialized: %s", self.__class__.__name__)

    def detect_faces(self, batch: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray]]:
        """Detects faces in an image, and returns bounding boxes and points for them.

        Parameters
        ----------
        batch: :class:`numpy.ndarray`
            The input batch of images to detect face in

        Returns
        -------
        List
            list of numpy arrays containing the bounding box and 5 point landmarks
            of detected faces
        """
        retval = []
        for img in batch:
            imgPad = pad_img(img)
            img180 = cv2.flip(imgPad, 0)
            img90 = cv2.transpose(imgPad)
            imgNeg90 = cv2.flip(img90, 0)

            winlist = stage1(img, imgPad, self.pcn1, classThreshold_[0])
            winlist = NMS(winlist, True, nmsThreshold_[0])
            winlist = stage2(imgPad, img180, self.pcn2, classThreshold_[1], 24, winlist)
            winlist = NMS(winlist, True, nmsThreshold_[1])
            winlist = stage3(imgPad, img180, img90, imgNeg90, self.pcn3, classThreshold_[2], 48, winlist)
            winlist = NMS(winlist, False, nmsThreshold_[2])
            winlist = deleteFP(winlist)
            winlist = trans_window(img, imgPad, winlist) # list of Window objects (faces)
        return winlist

class Window2:
    def __init__(self, x, y, w, h, angle, scale, conf):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.angle = angle
        self.scale = scale
        self.conf = conf


def preprocess_img(img, dim=None):
    if dim:
        img = cv2.resize(img, (dim, dim), interpolation=cv2.INTER_NEAREST)
    return img - np.array([104, 117, 123])

def resize_img(img, scale:float):
    h, w = img.shape[:2]
    h_, w_ = int(h / scale), int(w / scale)
    img = img.astype(np.float32) # fix opencv type error
    ret = cv2.resize(img, (w_, h_), interpolation=cv2.INTER_NEAREST)
    return ret

def pad_img(img:np.array):
    row = min(int(img.shape[0] * 0.2), 100)
    col = min(int(img.shape[1] * 0.2), 100)
    ret = cv2.copyMakeBorder(img, row, row, col, col, cv2.BORDER_CONSTANT)
    return ret

def legal(x, y, img):
    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
        return True
    else:
        return False

def inside(x, y, rect:Window2):
    if rect.x <= x < (rect.x + rect.w) and rect.y <= y < (rect.y + rect.h):
        return True
    else:
        return False

def smooth_angle(a, b):
    if a > b:
        a, b = b, a
    diff = (b - a) % 360
    if diff < 180:
        return a + diff // 2
    else:
        return b + (360 - diff) // 2

# use global variable `prelist` to mimic static variable in C++
prelist = []
def smooth_window(winlist):
    global prelist
    for win in winlist:
        for pwin in prelist:
            if IoU(win, pwin) > 0.9:
                win.conf = (win.conf + pwin.conf) / 2
                win.x = pwin.x
                win.y = pwin.y
                win.w = pwin.w
                win.h = pwin.h
                win.angle = pwin.angle
            elif IoU(win, pwin) > 0.6:
                win.conf = (win.conf + pwin.conf) / 2
                win.x = (win.x + pwin.x) // 2
                win.y = (win.y + pwin.y) // 2
                win.w = (win.w + pwin.w) // 2
                win.h = (win.h + pwin.h) // 2
                win.angle = smooth_angle(win.angle, pwin.angle)
    prelist = winlist
    return winlist

def IoU(w1:Window2, w2:Window2) -> float:
    xOverlap = max(0, min(w1.x + w1.w - 1, w2.x + w2.w - 1) - max(w1.x, w2.x) + 1)
    yOverlap = max(0, min(w1.y + w1.h - 1, w2.y + w2.h - 1) - max(w1.y, w2.y) + 1)
    intersection = xOverlap * yOverlap
    unio = w1.w * w1.h + w2.w * w2.h - intersection
    return intersection / unio

def NMS(winlist, local:bool, threshold:float):
    length = len(winlist)
    if length == 0:
        return winlist
    winlist.sort(key=lambda x: x.conf, reverse=True)
    flag = [0] * length
    for i in range(length):
        if flag[i]:
            continue
        for j in range(i+1, length):
            if local and abs(winlist[i].scale - winlist[j].scale) > EPS:
                continue
            if IoU(winlist[i], winlist[j]) > threshold:
                flag[j] = 1
    ret = [winlist[i] for i in range(length) if not flag[i]]
    return ret

def deleteFP(winlist):
    length = len(winlist)
    if length == 0:
        return winlist
    winlist.sort(key=lambda x: x.conf, reverse=True)
    flag = [0] * length
    for i in range(length):
        if flag[i]:
            continue
        for j in range(i+1, length):
            win = winlist[j]
            if inside(win.x, win.y, winlist[i]) and inside(win.x + win.w - 1, win.y + win.h - 1, winlist[i]):
                flag[j] = 1
    ret = [winlist[i] for i in range(length) if not flag[i]]
    return ret


# using if-else to mimic method overload in C++
def set_input(img):
    if type(img) == list:
        img = np.stack(img, axis=0)
    else:
        img = img[np.newaxis, :, :, :]
    #img = img.transpose((0, 3, 1, 2))
    return img


def trans_window(img, imgPad, winlist):
    """transfer Window2 to Window1 in winlist"""
    row = (imgPad.shape[0] - img.shape[0]) // 2
    col = (imgPad.shape[1] - img.shape[1]) // 2
    ret = list()
    for win in winlist:
        if win.w > 0 and win.h > 0:
            ret.append(Window(win.x-col, win.y-row, win.w, win.angle, win.conf))
    return ret


def stage1(img, imgPad, net, thres):
    row = (imgPad.shape[0] - img.shape[0]) // 2
    col = (imgPad.shape[1] - img.shape[1]) // 2
    winlist = []
    netSize = 24
    curScale = minFace_ / netSize
    img_resized = resize_img(img, curScale)
    while min(img_resized.shape[:2]) >= netSize:
        net_input = preprocess_img(img_resized)
        # net forward
        cls_prob, rotate, bbox = net.predict(net_input)

        w = netSize * curScale
        for i in range(cls_prob.shape[2]): # cls_prob[2]->height
            for j in range(cls_prob.shape[3]): # cls_prob[3]->width
                if cls_prob[0, 1, i, j].item() > thres:
                    sn = bbox[0, 0, i, j].item()
                    xn = bbox[0, 1, i, j].item()
                    yn = bbox[0, 2, i, j].item()
                    rx = int(j * curScale * stride_ - 0.5 * sn * w + sn * xn * w + 0.5 * w) + col
                    ry = int(i * curScale * stride_ - 0.5 * sn * w + sn * yn * w + 0.5 * w) + row
                    rw = int(w * sn)
                    if legal(rx, ry, imgPad) and legal(rx + rw - 1, ry + rw -1, imgPad):
                        if rotate[0, 1, i, j].item() > 0.5:
                            winlist.append(Window2(rx, ry, rw, rw, 0, curScale, cls_prob[0, 1, i, j].item()))
                        else:
                            winlist.append(Window2(rx, ry, rw, rw, 180, curScale, cls_prob[0, 1, i, j].item()))
        img_resized = resize_img(img_resized, scale_)
        curScale = img.shape[0] / img_resized.shape[0]
    return winlist

def stage2(img, img180, net, thres, dim, winlist):
    length = len(winlist)
    if length == 0:
        return winlist
    datalist = []
    height = img.shape[0]
    for win in winlist:
        if abs(win.angle) < EPS:
            datalist.append(preprocess_img(img[win.y:win.y+win.h, win.x:win.x+win.w, :], dim))
        else:
            y2 = win.y + win.h -1
            y = height - 1 - y2
            datalist.append(preprocess_img(img180[y:y+win.h, win.x:win.x+win.w, :], dim))
    # net forward
    cls_prob, rotate, bbox = net.predict(datalist)

    ret = []
    for i in range(length):
        if cls_prob[i, 1].item() > thres:
            sn = bbox[i, 0].item()
            xn = bbox[i, 1].item()
            yn = bbox[i, 2].item()
            cropX = winlist[i].x
            cropY = winlist[i].y
            cropW = winlist[i].w
            if abs(winlist[i].angle) > EPS:
                cropY = height - 1 - (cropY + cropW - 1)
            w = int(sn * cropW)
            x = int(cropX - 0.5 * sn * cropW + cropW * sn * xn + 0.5 * cropW)
            y = int(cropY - 0.5 * sn * cropW + cropW * sn * yn + 0.5 * cropW)
            maxRotateScore = 0
            maxRotateIndex = 0
            for j in range(3):
                if rotate[i, j].item() > maxRotateScore:
                    maxRotateScore = rotate[i, j].item()
                    maxRotateIndex = j
            if legal(x, y, img) and legal(x+w-1, y+w-1, img):
                angle = 0
                if abs(winlist[i].angle) < EPS:
                    if maxRotateIndex == 0:
                        angle = 90
                    elif maxRotateIndex == 1:
                        angle = 0
                    else:
                        angle = -90
                    ret.append(Window2(x, y, w, w, angle, winlist[i].scale, cls_prob[i, 1].item()))
                else:
                    if maxRotateIndex == 0:
                        angle = 90
                    elif maxRotateIndex == 1:
                        angle = 180
                    else:
                        angle = -90
                    ret.append(Window2(x, height-1-(y+w-1), w, w, angle, winlist[i].scale, cls_prob[i, 1].item()))
    return ret

def stage3(imgPad, img180, img90, imgNeg90, net, thres, dim, winlist):
    length = len(winlist)
    if length == 0:
        return winlist

    datalist = []
    height, width = imgPad.shape[:2]

    for win in winlist:
        if abs(win.angle) < EPS:
            datalist.append(preprocess_img(imgPad[win.y:win.y+win.h, win.x:win.x+win.w, :], dim))
        elif abs(win.angle - 90) < EPS:
            datalist.append(preprocess_img(img90[win.x:win.x+win.w, win.y:win.y+win.h, :], dim))
        elif abs(win.angle + 90) < EPS:
            x = win.y
            y = width - 1 - (win.x + win.w -1)
            datalist.append(preprocess_img(imgNeg90[y:y+win.h, x:x+win.w, :], dim))
        else:
            y2 = win.y + win.h - 1
            y = height - 1 - y2
            datalist.append(preprocess_img(img180[y:y+win.h, win.x:win.x+win.w], dim))
    # network forward
    cls_prob, rotate, bbox = net.predict(datalist)

    ret = []
    for i in range(length):
        if cls_prob[i, 1].item() > thres:
            sn = bbox[i, 0].item()
            xn = bbox[i, 1].item()
            yn = bbox[i, 2].item()
            cropX = winlist[i].x
            cropY = winlist[i].y
            cropW = winlist[i].w
            img_tmp = imgPad
            if abs(winlist[i].angle - 180) < EPS:
                cropY = height - 1 - (cropY + cropW -1)
                img_tmp = img180
            elif abs(winlist[i].angle - 90) < EPS:
                cropX, cropY = cropY, cropX
                img_tmp = img90
            elif abs(winlist[i].angle + 90) < EPS:
                cropX = winlist[i].y
                cropY = width -1 - (winlist[i].x + winlist[i].w - 1)
                img_tmp = imgNeg90

            w = int(sn * cropW)
            x = int(cropX - 0.5 * sn * cropW + cropW * sn * xn + 0.5 * cropW)
            y = int(cropY - 0.5 * sn * cropW + cropW * sn * yn + 0.5 * cropW)
            angle = angleRange_ * rotate[i, 0].item()
            if legal(x, y, img_tmp) and legal(x+w-1, y+w-1, img_tmp):
                if abs(winlist[i].angle) < EPS:
                    ret.append(Window2(x, y, w, w, angle, winlist[i].scale, cls_prob[i, 1].item()))
                elif abs(winlist[i].angle - 180) < EPS:
                    ret.append(Window2(x, height-1-(y+w-1), w, w, 180-angle, winlist[i].scale, cls_prob[i, 1].item()))
                elif abs(winlist[i].angle - 90) < EPS:
                    ret.append(Window2(y, x, w, w, 90-angle, winlist[i].scale, cls_prob[i, 1].item()))
                else:
                    ret.append(Window2(width-y-w, x, w, w, -90+angle, winlist[i].scale, cls_prob[i, 1].item()))
    return ret
