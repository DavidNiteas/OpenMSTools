import os
from typing import Literal
import logging

import dask.bag as db
from pydantic import Field

from ..ms_tools import (
    AdductDetector,
    AdductDetectorConfig,
    Centrizer,
    CentrizerConfig,
    FeatureFinder,
    FeatureFinderConfig,
    FeatureLinker,
    FeatureLinkerConfig,
    MSTool,
    MSToolConfig,
    RTAligner,
    RTAlignerConfig,
    TICSmoother,
    TICSmootherConfig,
)
from ..snaps.MassDataModule.data_module import ConsensusMap, FeatureMap, SpectrumMap, XICMap, link_ms2_and_feature_map
from ..snaps.MassDataModule.data_module.data_wrapper import MetaMSDataWrapper, OpenMSDataWrapper
from .reference_analysis import ReferenceFeatureFinder, ReferenceFeatureFinderConfig


class ExperimentAnalysisConfig(MSToolConfig):

    use_tic_smoother: bool = Field(
        default=True,
        description="是否使用TIC平滑"
    )
    tic_smoother_config: TICSmootherConfig = Field(
        default=TICSmootherConfig(),
        description="TIC平滑的配置"
    )
    use_centrizer: bool = Field(
        default=False,
        description="是否使用质心化校正"
    )
    centrizer_config: CentrizerConfig = Field(
        default=CentrizerConfig(),
        description="质心化校正的配置"
    )
    feature_finder_config: FeatureFinderConfig = Field(
        default=FeatureFinderConfig(),
        description="特征提取的配置"
    )
    use_rt_aligner: bool = Field(
        default=True,
        description="是否使用RT对齐"
    )
    rt_aligner_config: RTAlignerConfig = Field(
        default=RTAlignerConfig(),
        description="RT对齐的配置"
    )
    reference_analysis_config: ReferenceFeatureFinderConfig = Field(
        default=ReferenceFeatureFinderConfig(),
        description="参考文件的分析配置，用于RT对齐"
    )
    use_adduct_detector: bool = Field(
        default=True,
        description="是否使用加合物检测"
    )
    adduct_detector_config: AdductDetectorConfig = Field(
        default=AdductDetectorConfig(),
        description="加合物检测的配置"
    )
    use_feature_linker: bool = Field(
        default=True,
        description="是否使用特征链接（如果只有一个输入文件，则永远不会进行特征链接）"
    )
    feature_linker_config: FeatureLinkerConfig = Field(
        default=FeatureLinkerConfig(),
        description="特征链接的配置（如果只有一个输入文件，则不会进行特征链接）"
    )
    worker_type: Literal["processes", "threads", "synchronous"] = Field(
        default="processes",
        description="使用的工作类型，\
            `processes`表示使用多进程，\
            `threads`表示使用多线程 \
            `synchronous`表示使用单线程同步调度，在调式时非常有用"
    )
    num_workers: int = Field(
        default=os.cpu_count() if os.cpu_count() < 61 else 61,
        description="并行计算的线程数/进程数"
    )
    
class ExperimentAnalysisWorkerError(Exception):
    
    def __init__(self, message, file_path, worker_error=None):
        super().__init__(message)
        self.file_path = file_path
        self.worker_error = worker_error

    def __str__(self):
        error_message = f"ExperimentAnalysisWorkerError: {self.args[0]}"
        if self.worker_error:
            error_message += f"\Worker error: {self.worker_error}"
        error_message += f"\nFile path: {self.file_path}"
        return error_message
    
class ExperimentAnalysisError(Exception):
    
    def __init__(self, worker_errors):
        super().__init__("Experiment analysis encountered one or more worker errors.")
        self.worker_errors = worker_errors

    def __str__(self):
        error_message = "ExperimentAnalysisError: Experiment analysis encountered one or more worker errors.\n"
        for error in self.worker_errors:
            error_message += f" - {error}\n"
        return error_message.strip()
    
    @classmethod
    def check(cls, worker_outputs: list[MetaMSDataWrapper | ExperimentAnalysisWorkerError]):
        worker_errors = [
            worker_output \
                for worker_output in worker_outputs \
                if isinstance(worker_output, ExperimentAnalysisWorkerError)
        ]
        if len(worker_errors) > 0:
            raise cls(worker_errors)
        
def partition_list(input_list, partition_size):
    """
    对列表进行分区，并返回固定元素数量的列表切片。

    :param input_list: 要分区的列表
    :param partition_size: 每个分区的大小
    :return: 包含分区列表的列表
    """
    if partition_size <= 0:
        raise ValueError("分区大小必须大于0")
    
    partitions = []
    for i in range(0, len(input_list), partition_size):
        partition = input_list[i:i + partition_size]
        partitions.append(partition)
    
    return partitions

class ExperimentAnalysis(MSTool):

    config_type = ExperimentAnalysisConfig
    config: ExperimentAnalysisConfig

    def _single_file_pipeline(
        self,
        exp_file_path: str,
        ref: ConsensusMap | FeatureMap | None = None,
    ) -> MetaMSDataWrapper | ExperimentAnalysisWorkerError:
        try:
            open_ms_wrapper = OpenMSDataWrapper(file_paths=[exp_file_path])
            open_ms_wrapper.init_exps()
            if self.config.use_tic_smoother:
                open_ms_wrapper = TICSmoother(config=self.config.tic_smoother_config)(open_ms_wrapper)
            if self.config.use_centrizer:
                open_ms_wrapper = Centrizer(config=self.config.centrizer_config)(open_ms_wrapper)
            open_ms_wrapper = FeatureFinder(config=self.config.feature_finder_config)(open_ms_wrapper)
            if self.config.use_rt_aligner and ref is not None:
                ref_map = ref.get_oms_feature_map()
                open_ms_wrapper.ref_feature_for_align = ref_map
                open_ms_wrapper = RTAligner(config=self.config.rt_aligner_config)(open_ms_wrapper)
            if self.config.use_adduct_detector:
                open_ms_wrapper = AdductDetector(config=self.config.adduct_detector_config)(open_ms_wrapper)
            meta_ms_wrapper = MetaMSDataWrapper(
                file_paths=open_ms_wrapper.file_paths,
                exp_names=open_ms_wrapper.exp_names,
                spectrum_maps=[SpectrumMap.from_oms(open_ms_wrapper.exps[0], open_ms_wrapper.exp_names[0])],
                xic_maps=[XICMap.from_oms(open_ms_wrapper.exps[0])],
                feature_maps=[FeatureMap.from_oms(
                    open_ms_wrapper.features[0],
                    open_ms_wrapper.chromatogram_peaks[0],
                    open_ms_wrapper.exp_names[0]
                )],
            )
            meta_ms_wrapper.ms2_feature_mapping = link_ms2_and_feature_map(
                meta_ms_wrapper.feature_maps[0], meta_ms_wrapper.spectrum_maps[0], "spectrum"
            )
            return meta_ms_wrapper
        except Exception as e:
            return ExperimentAnalysisWorkerError(f"Worker处理文件时出错: {exp_file_path}", exp_file_path, str(e))

    def __call__(
        self,
        exp_file_paths: list[str],
        ref_file_paths: list[str] | None = None,
    ) -> MetaMSDataWrapper:
        if len(exp_file_paths) > 1 and self.config.use_feature_linker:
            if ref_file_paths is None:
                ref_file_paths = exp_file_paths
            ref = ReferenceFeatureFinder(config=self.config.reference_analysis_config)(ref_file_paths)
        else:
            ref = None
        single_file_bag = db.from_sequence(
            zip(exp_file_paths, [ref] * len(exp_file_paths))
        )
        single_file_bag = single_file_bag.map(lambda x: self._single_file_pipeline(x[0], x[1]))
        meta_ms_wrapper_list = single_file_bag.compute(
            scheduler=self.config.worker_type,
            num_workers=self.config.num_workers,
        )
        ExperimentAnalysisError.check(meta_ms_wrapper_list)
        meta_ms_wrapper = MetaMSDataWrapper.merge(meta_ms_wrapper_list)
        if self.config.use_feature_linker and len(exp_file_paths) > 1:
            feature_list = [feature_map.get_oms_feature_map() for feature_map in meta_ms_wrapper.feature_maps]
            open_ms_wrapper = OpenMSDataWrapper(
                file_paths=meta_ms_wrapper.file_paths,
                exp_names=meta_ms_wrapper.exp_names,
                features=feature_list,
            )
            open_ms_wrapper = FeatureLinker(self.config.feature_linker_config)(open_ms_wrapper)
            meta_ms_wrapper.consensus_map = ConsensusMap.from_oms(open_ms_wrapper.consensus_map)
        return meta_ms_wrapper
    
    def _debug(
        self,
        exp_file_paths: list[str],
        ref_file_paths: list[str] | None = None,
    ) -> MetaMSDataWrapper:
        if len(exp_file_paths) > 1 and self.config.use_feature_linker:
            if ref_file_paths is None:
                ref_file_paths = exp_file_paths
            ref = ReferenceFeatureFinder(config=self.config.reference_analysis_config)._debug(ref_file_paths)
        else:
            ref = None
        meta_ms_wrapper_list = []
        for batch in partition_list(exp_file_paths, self.config.num_workers):
            single_file_bag = db.from_sequence(
                zip(batch, [ref] * len(batch))
            )
            single_file_bag = single_file_bag.map(lambda x: self._single_file_pipeline(x[0], x[1]))
            batch_results = single_file_bag.compute(
                scheduler=self.config.worker_type,
                num_workers=self.config.num_workers,
            )
            ExperimentAnalysisError.check(batch_results)
            meta_ms_wrapper_list += batch_results
            logging.debug(f"完成了{len(batch_results)}个文件,没有发生错误")
        meta_ms_wrapper = MetaMSDataWrapper.merge(meta_ms_wrapper_list)
        if self.config.use_feature_linker and len(exp_file_paths) > 1:
            feature_list = [feature_map.get_oms_feature_map() for feature_map in meta_ms_wrapper.feature_maps]
            open_ms_wrapper = OpenMSDataWrapper(
                file_paths=meta_ms_wrapper.file_paths,
                exp_names=meta_ms_wrapper.exp_names,
                features=feature_list,
            )
            open_ms_wrapper = FeatureLinker(self.config.feature_linker_config)(open_ms_wrapper)
            meta_ms_wrapper.consensus_map = ConsensusMap.from_oms(open_ms_wrapper.consensus_map)
        return meta_ms_wrapper
