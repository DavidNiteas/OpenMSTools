from typing import ClassVar, Literal, Optional

import dask.bag as db
import pyopenms as oms
from pydantic import Field

from ...snaps.MassDataModule.data_module.configs import OpenMSMethodConfig
from ...snaps.MassDataModule.data_module.wrappers import OpenMSDataWrapper
from ..ABCs import MSTool


class TICSmootherConfig(OpenMSMethodConfig):

    openms_method: ClassVar[type[oms.SavitzkyGolayFilter]] = oms.SavitzkyGolayFilter
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

class TICSmoother(MSTool):

    config_type = TICSmootherConfig
    config: TICSmootherConfig

    def __init__(self, config: Optional[TICSmootherConfig] = None):
        super().__init__(config)
        self.smoother = oms.SavitzkyGolayFilter()

    def __call__(self, data: OpenMSDataWrapper, **kwargs) -> OpenMSDataWrapper:
        runtime_config = self.config.get_runtime_config(** kwargs)
        self.smoother.setParameters(runtime_config.param)

        if len(data.exps) == 0:
            return data

        def run_smoothing(exp):
            self.smoother.filterExperiment(exp)
            return exp

        if len(data.exps) == 1:
            data.exps[0] = run_smoothing(data.exps[0])
        else:
            inputs_bag = db.from_sequence(data.exps, npartitions=runtime_config.num_workers)
            outputs_bag = inputs_bag.map(run_smoothing)
            data.exps = outputs_bag.compute(
                scheduler=runtime_config.worker_type,
                num_workers=runtime_config.num_workers
            )

        return data
