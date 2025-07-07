from typing import Literal

from pydantic import Field

from ..ms_tools import (
    ConsensusMap,
    FeatureFinder,
    FeatureFinderConfig,
    FeatureLinker,
    FeatureLinkerConfig,
    FeatureMap,
    MSTool,
    MSToolConfig,
    OpenMSDataWrapper,
)


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

class ReferenceFeatureFinder(MSTool):

    config_type = ReferenceFeatureFinderConfig
    config: ReferenceFeatureFinderConfig

    def __call__(
        self,
        ref_file_paths: list[str],
    ) -> FeatureMap | ConsensusMap:
        ref_datas = OpenMSDataWrapper(file_paths=ref_file_paths)
        ref_datas.init_exps()
        feature_finder = FeatureFinder(config=self.config.feature_finder_config)
        ref_datas = feature_finder(ref_datas)
        if len(ref_datas.features) > 1 and self.config.mode == "consensus":
            feature_linker = FeatureLinker(config=self.config.feature_linker_config)
            ref_datas = feature_linker(ref_datas.features)
        if ref_datas.consensus_map is not None:
            return ConsensusMap.from_oms(ref_datas.consensus_map)
        else:
            ref_datas.infer_ref_feature_for_align()
            return FeatureMap.from_oms(ref_datas.ref_feature_for_align)
