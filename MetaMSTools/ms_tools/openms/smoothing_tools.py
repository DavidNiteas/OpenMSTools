from typing import ClassVar, Optional

import dask.bag as db
import pyopenms as oms

from ..ABCs import MSTool, OpenMSDataWrapper, OpenMSMethodConfig


class TICSmootherConfig(OpenMSMethodConfig):

    openms_method: ClassVar[type[oms.SavitzkyGolayFilter]] = oms.SavitzkyGolayFilter

class TICSmoother(MSTool):

    config_type = TICSmootherConfig
    config: TICSmootherConfig

    def __init__(self, config: Optional[TICSmootherConfig] = None):
        super().__init__(config)
        self.smoother = oms.SavitzkyGolayFilter()

    def __call__(self, data: OpenMSDataWrapper) -> OpenMSDataWrapper:

        if len(data.exps) == 0:
            return data

        def run_smoothing(exp):
            self.smoother.filterExperiment(exp)
            return exp

        if len(data.exps) == 1:
            data.exps[0] = run_smoothing(data.exps[0])
        else:
            inputs_bag = db.from_sequence(data.exps)
            outputs_bag = inputs_bag.map(run_smoothing)
            data.exps = outputs_bag.compute(scheduler="threads")

        return data
