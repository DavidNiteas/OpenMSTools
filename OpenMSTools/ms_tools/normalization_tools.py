from typing import ClassVar, Literal

import dask.bag as db
import pyopenms as oms
from pydantic import Field

from .ABCs import MSTool, OpenMSDataWrapper, OpenMSMethodConfig


class SpectrumNormalizerConfig(OpenMSMethodConfig):

    openms_method: ClassVar[type[oms.Normalizer]] = oms.Normalizer

    method: Literal["to_one","to_TIC"] = Field(
        default="to_one",
        description="通过将每个光谱除以其TIC（总离子流）进行归一化（‘to_TIC’）或将其归一化到单个光谱的最大强度（‘to_one’）。"
    )

class SpectrumNormalizer(MSTool):

    config_type = SpectrumNormalizerConfig
    config: SpectrumNormalizerConfig

    def __init__(self, config = None):
        super().__init__(config)
        self.normalizer = oms.Normalizer()

    def __call__(self, data: OpenMSDataWrapper) -> OpenMSDataWrapper:

        self.normalizer.setParameters(self.config.param)

        if len(data.exps) == 0:
            return data

        def run_normalization(exp):
            self.normalizer.filterPeakMap(exp)
            return exp

        if len(data.exps) == 1:
            data.exps[0] = [run_normalization(data.exps[0])]
        else:
            inputs_bag = db.from_sequence(data.exps)
            outputs_bag = inputs_bag.map(run_normalization)
            data.exps = outputs_bag.compute(scheduler="threads")

        return data
