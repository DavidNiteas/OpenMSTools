from typing import ClassVar, Literal

import dask.bag as db
import pyopenms as oms
from pydantic import Field

from ...snaps.MassDataModule.data_module.configs import OpenMSMethodConfig
from ...snaps.MassDataModule.data_module.wrappers import OpenMSDataWrapper
from ..ABCs import MSTool


class SpectrumNormalizerConfig(OpenMSMethodConfig):

    openms_method: ClassVar[type[oms.Normalizer]] = oms.Normalizer

    method: Literal["to_one","to_TIC"] = Field(
        default="to_one",
        description="通过将每个光谱除以其TIC（总离子流）进行归一化（‘to_TIC’）或将其归一化到单个光谱的最大强度（‘to_one’）。"
    )
    worker_type: Literal['threads','synchronous'] = Field(
        default='threads',
        is_openms_method_param=False,
        description="工作模式。\
            'threads'使用线程池，'synchronous'使用单线程。"
    )
    num_workers: int | None = Field(
        default=None,
        is_openms_method_param=False,
        description="并行worker的数量。\
            如果为None，则使用所有可用的CPU核心。"
    )

class SpectrumNormalizer(MSTool):

    config_type = SpectrumNormalizerConfig
    config: SpectrumNormalizerConfig

    def __init__(self, config = None):
        super().__init__(config)
        self.normalizer = oms.Normalizer()

    def __call__(self, data: OpenMSDataWrapper, **kwargs) -> OpenMSDataWrapper:
        runtime_config = self.config.get_runtime_config(** kwargs)
        self.normalizer.setParameters(runtime_config.param)

        if len(data.exps) == 0:
            return data

        def run_normalization(exp):
            self.normalizer.filterPeakMap(exp)
            return exp

        if len(data.exps) == 1:
            data.exps[0] = [run_normalization(data.exps[0])]
        else:
            inputs_bag = db.from_sequence(data.exps, npartitions=runtime_config.num_workers)
            outputs_bag = inputs_bag.map(run_normalization)
            data.exps = outputs_bag.compute(
                scheduler=runtime_config.worker_type,
                num_workers=runtime_config.num_workers
            )

        return data
