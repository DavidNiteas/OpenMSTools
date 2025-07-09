import os
from typing import Literal

import dask.bag as db
from pydantic import Field

from ..ms_tools import (
    FeatureFinder,
    FeatureFinderConfig,
    FeatureLinker,
    FeatureLinkerConfig,
    MSTool,
    MSToolConfig,
)
from ..snaps.MassDataModule.data_module.data_wrapper import OpenMSDataWrapper
from ..snaps.MassDataModule.data_module.experiment_module import ConsensusMap, FeatureMap


class ReferenceFeatureFinderConfig(MSToolConfig):

    feature_finder_config: FeatureFinderConfig = Field(
        default=FeatureFinderConfig(),
        description="用于检测参考样本中特征的配置参数",
    )
    feature_linker_config: FeatureLinkerConfig = Field(
        default=FeatureLinkerConfig(),
        description="用于链接参考样本中特征的配置参数\
            如果有多个参考样本且`mode`设置为`consensus`，会使用这一部分参数",
    )
    mode: Literal["consensus","max_size_feature"] = Field(
        default="consensus",
        description="参考特征的选择模式，\
            `consensus`表示使用共识特征，\
            `max_size_feature`表示使用特征数最多的样本中的特征图 \
            如果只有一个参考样本，则无论mode如何，都会使用该样本的特征",
    )
    worker_type: Literal["processes", "threads", "synchronous"] = Field(
        default="processes",
        description="使用的工作类型，\
            `processes`表示使用多进程，\
            `threads`表示使用多线程 \
            `synchronous`表示使用单线程同步调度，在调式时非常有用"
    )
    num_workers: int = Field(
        default=os.cpu_count(),
        description="并行计算的线程数/进程数"
    )

class ReferenceFeatureFinder(MSTool):

    config_type = ReferenceFeatureFinderConfig
    config: ReferenceFeatureFinderConfig

    def _single_file_pipeline(
        self,
        ref_file_path: str,
    ) -> FeatureMap:
        ref_datas = OpenMSDataWrapper(file_paths=[ref_file_path])
        ref_datas.init_exps()
        feature_finder = FeatureFinder(config=self.config.feature_finder_config)
        ref_datas = feature_finder(ref_datas)
        return FeatureMap.from_oms(ref_datas.features[0],ref_datas.chromatogram_peaks[0],ref_datas.exp_names[0])

    def __call__(
        self,
        ref_file_paths: list[str],
    ) -> FeatureMap | ConsensusMap:
        ref_file_path_bag = db.from_sequence(ref_file_paths,npartitions=self.config.num_workers)
        ref_feature_map_bag = ref_file_path_bag.map(self._single_file_pipeline)
        ref_feature_map:list[FeatureMap] = ref_feature_map_bag.compute(scheduler=self.config.worker_type)
        ref_datas = [ref.get_oms_feature_map() for ref in ref_feature_map]
        ref_datas = OpenMSDataWrapper(
            exp_names=ref_file_paths,
            features=ref_datas
        )
        if len(ref_datas.features) > 1 and self.config.mode == "consensus":
            feature_linker = FeatureLinker(config=self.config.feature_linker_config)
            ref_datas = feature_linker(ref_datas)
        if ref_datas.consensus_map is not None:
            return ConsensusMap.from_oms(ref_datas.consensus_map)
        else:
            ref_datas.infer_ref_feature_for_align()
            return FeatureMap.from_oms(ref_datas.ref_feature_for_align)
