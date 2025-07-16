from typing import Literal

import pyopenms as oms
from pydantic import Field

from ..ms_tools import (
    FeatureFinder,
    FeatureFinderConfig,
    MSTool,
    MSToolConfig,
)
from ..snaps.MassDataModule.data_module.wrappers import OpenMSDataWrapper


class ReferenceFeatureFinderConfig(MSToolConfig):

    feature_finder_config: FeatureFinderConfig = Field(
        default=FeatureFinderConfig(),
        description="用于检测参考样本中特征的配置参数",
    )
    mode: Literal["mix","max_size_feature"] = Field(
        default="mix",
        description="参考特征的选择模式，\
            `mix`表示使用混合样本，\
            `max_size_feature`表示使用特征数最多的样本 \
            如果只有一个参考样本，则无论mode如何，都会使用该样本的特征",
    )
    worker_type: Literal["threads", "synchronous"] = Field(
        default="threads",
        description="使用的工作类型，\
            `threads`表示使用多线程 \
            `synchronous`表示使用单线程同步调度，在调式时非常有用"
    )
    num_workers: int | None = Field(
        default=None,
        description="并行计算的线程数/进程数"
    )

class ReferenceFeatureFinder(MSTool):

    config_type = ReferenceFeatureFinderConfig
    config: ReferenceFeatureFinderConfig

    @staticmethod
    def _single_func(
        runtime_config: ReferenceFeatureFinderConfig,
        data_wrapper: OpenMSDataWrapper,
        result_type: Literal['feature', 'experiment'] = 'feature'
    ) -> OpenMSDataWrapper:
        match result_type:
            case 'feature':
                data_wrapper = FeatureFinder(runtime_config.feature_finder_config)(
                    data_wrapper,
                    worker_type = runtime_config.worker_type,
                    num_workers = runtime_config.num_workers,
                )
                data_wrapper.ref_for_align = data_wrapper.features[0]
            case'experiment':
                data_wrapper.ref_for_align = data_wrapper.exps[0]
        return data_wrapper

    @staticmethod
    def _max_size_feature_func(
        runtime_config: ReferenceFeatureFinderConfig,
        data_wrapper: OpenMSDataWrapper,
        result_type: Literal['feature', 'experiment'] = 'feature'
    ) -> OpenMSDataWrapper:
        data_wrapper = FeatureFinder(runtime_config.feature_finder_config)(
            data_wrapper,
            worker_type = runtime_config.worker_type,
            num_workers = runtime_config.num_workers,
        )
        max_idx = None
        max_size = 0
        for i, feature_map in enumerate(data_wrapper.features):
            if feature_map.size() > max_size:
                max_idx = i
                max_size = feature_map.size()
        match result_type:
            case 'feature':
                data_wrapper.ref_for_align = data_wrapper.features[max_idx]
            case'experiment':
                data_wrapper.ref_for_align = data_wrapper.exps[max_idx]
        return data_wrapper

    @staticmethod
    def _mix_func(
        runtime_config: ReferenceFeatureFinderConfig,
        data_wrapper: OpenMSDataWrapper,
        result_type: Literal['feature', 'experiment'] = 'feature'
    ) -> OpenMSDataWrapper:
        mix_exp = oms.MSExperiment()
        for exp in data_wrapper.exps:
            for spectrum in exp:
                mix_exp.addSpectrum(spectrum)
        mix_data_wrapper = OpenMSDataWrapper(
            exps=[mix_exp],
        )
        match result_type:
            case 'feature':
                mix_data_wrapper = FeatureFinder(runtime_config.feature_finder_config)(
                    mix_data_wrapper,
                    worker_type = runtime_config.worker_type,
                    num_workers = runtime_config.num_workers,
                )
                data_wrapper.ref_for_align = mix_data_wrapper.features[0]
            case'experiment':
                data_wrapper.ref_for_align = mix_exp
        return data_wrapper

    def __call__(
        self,
        ref_file_paths: list[str],
        *args,
        result_type: Literal['feature', 'experiment'] = 'feature',
        **kwargs,
    ) -> oms.FeatureMap | oms.MSExperiment:

        runtime_config = self.config.get_runtime_config(**kwargs)

        data_wrapper = OpenMSDataWrapper(
            file_paths=ref_file_paths,
        )
        data_wrapper.init_exps(
            worker_type=runtime_config.worker_type,
            num_workers=runtime_config.num_workers,
        )
        if len(data_wrapper.exps) > 1:
            match runtime_config.mode:
                case'max_size_feature':
                    data_wrapper = self._max_size_feature_func(runtime_config, data_wrapper, result_type)
                case'mix':
                    data_wrapper = self._mix_func(runtime_config, data_wrapper, result_type)
        else:
            data_wrapper = self._single_func(runtime_config, data_wrapper, result_type)
        return data_wrapper.ref_for_align
