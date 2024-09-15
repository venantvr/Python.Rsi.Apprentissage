from typing import Callable

# noinspection PyUnresolvedReferences
import psutil
from pandas import DataFrame

from apprentissage.bootstrap.transform_result import TransformResult

MlTranformLambda = Callable[[DataFrame], TransformResult]
HyperoptLambda = Callable[[dict], float | int]
