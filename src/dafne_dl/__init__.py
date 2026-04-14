#  Copyright (c) 2021 Dafne-Imaging Team
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

# -*- coding: utf-8 -*-

VERSION='2.0.1-beta2'
__version__ = VERSION

from . import resources

import sys
import flexidep

assert sys.version_info.major == 3, "This software is only compatible with Python 3.x"

if sys.version_info.minor < 10:
    import importlib_resources as pkg_resources
else:
    import importlib.resources as pkg_resources

# install the required resources
if not flexidep.is_frozen():
    with pkg_resources.files(resources).joinpath('runtime_dependencies.cfg').open() as f:
        dm = flexidep.DependencyManager(config_file=f)
    dm.install_auto()

from .DynamicDLModel import DynamicDLModel
from .DynamicEnsembleModel import DynamicEnsembleModel
from .DynamicTorchModel import DynamicTorchModel
from .LocalModelProvider import LocalModelProvider
from .RemoteModelProvider import RemoteModelProvider