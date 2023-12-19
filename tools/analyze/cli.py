#!/usr/bin/env python3
""" Command Line Arguments for tools """
import gettext

from lib.cli.args import FaceSwapArgs
from lib.cli.actions import DirFullPaths, SaveFileFullPaths, Radio, Slider


# LOCALES
_LANG = gettext.translation("tools.sort.cli", localedir="locales", fallback=True)
_ = _LANG.gettext


_HELPTEXT = _("This command lets you sort images using various methods.")

class AnalyzeArgs(FaceSwapArgs):
    """ Class to parse the command line arguments for sort tool """

    @staticmethod
    def get_info():
        """ Return command information """
        return _("Sort faces using a number of different techniques")

    @staticmethod
    def get_argument_list():
        """ Put the arguments in a list so that they are accessible from both argparse and gui """
        argument_list = []
        argument_list.append(dict(
            opts=('-i', '--input'),
            action=DirFullPaths,
            dest="input_dir",
            group=_("data"),
            help=_("Input directory of aligned faces."),
            required=True))
        argument_list.append(dict(
            opts=("-B", "--batch-mode"),
            action="store_true",
            dest="batch_mode",
            default=False,
            group=_("data"),
            help=_("R|If selected then the input_dir should be a parent folder containing "
                   "multiple folders of faces you wish to sort. The faces "
                   "will be output to separate sub-folders in the output_dir")))

        return argument_list
