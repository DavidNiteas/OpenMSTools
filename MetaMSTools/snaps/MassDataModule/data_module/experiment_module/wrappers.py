from __future__ import annotations

import json
import os
from typing import ClassVar, Literal

import dask
import dask.bag as db
import polars as pl
from pydantic import Field
from typing_extensions import Self

from ..module_abc import BaseWrapper
from .consensus import ConsensusMap
from .features import FeatureMap
from .spectrums import SpectrumMap
from .xic import XICMap


class MetaMSDataWrapper(BaseWrapper):

    _column_attributes: ClassVar[list[str]] = [
        "file_paths",
        "exp_names",
        "spectrum_maps",
        "xic_maps",
        "feature_maps",
    ]

    queue_name: str | None = Field(
        None,
        description="队列名称"
    )
    file_paths: list[str] | None = Field(
        None,
        description="原始文件路径列表"
    )
    exp_names: pl.Series | None = Field(
        None,
        description="实验名称列表"
    )
    spectrum_maps: list[SpectrumMap] | None = Field(
        None,
        description="谱图数据列表"
    )
    xic_maps: list[XICMap] | None = Field(
        None,
        description="XIC数据列表"
    )
    feature_maps: list[FeatureMap] | None = Field(
        None,
        description="特征图数据列表"
    )
    consensus_map: ConsensusMap | None = Field(
        None,
        description="共识特征图数据"
    )

    def save_metadata(self, save_dir_path: str) -> None:

        os.makedirs(save_dir_path, exist_ok=True)

        metadata = {
            "queue_name": self.queue_name,
            "file_paths": self.file_paths,
            "exp_names": self.exp_names.to_list()
        }
        with open(os.path.join(save_dir_path, "metadata.json"), "w") as f:
            json.dump(metadata, f)

    def save_consensus_map(self, save_dir_path: str) -> None:

        os.makedirs(save_dir_path, exist_ok=True)
        exp_data_dir = os.path.join(save_dir_path, "exp_datas")
        os.makedirs(exp_data_dir, exist_ok=True)

        consensus_map_save_path = os.path.join(exp_data_dir, "consensus_map")
        if self.consensus_map is not None:
            self.consensus_map.save(consensus_map_save_path)

    def save(
        self,
        save_dir_path: str,
        worker_type: Literal["threads", "processes", "synchronous"] = "threads",
        num_workers: int | None = None,
    ) -> None:

        os.makedirs(save_dir_path, exist_ok=True)

        if self.queue_name is not None:
            self.save_metadata(save_dir_path)

        exp_data_dir = os.path.join(save_dir_path, "exp_datas")
        os.makedirs(exp_data_dir, exist_ok=True)
        save_queue = []
        for i in range(self.size()):
            if self.exp_names is not None:
                exp_dir = os.path.join(exp_data_dir, self.exp_names[i])
            else:
                exp_dir = os.path.join(exp_data_dir, str(i))
            os.makedirs(exp_dir, exist_ok=True)
            if self.spectrum_maps is not None:
                spectrum_map_save_path = os.path.join(exp_dir, "spectrum_map")
                spectrum_map = self.spectrum_maps[i]
                if spectrum_map is not None:
                    save_queue.append((spectrum_map,spectrum_map_save_path))
            if self.xic_maps is not None:
                xic_map_save_path = os.path.join(exp_dir, "xic_map")
                xic_map = self.xic_maps[i]
                if xic_map is not None:
                    save_queue.append((xic_map,xic_map_save_path))
            if self.feature_maps is not None:
                feature_map_save_path = os.path.join(exp_dir, "feature_map")
                feature_map = self.feature_maps[i]
                if feature_map is not None:
                    save_queue.append((feature_map,feature_map_save_path))
        if self.consensus_map is not None:
            consensus_map_save_path = os.path.join(exp_data_dir, "consensus_map")
            consensus_map = self.consensus_map
            save_queue.append((consensus_map,consensus_map_save_path))
        save_bag = db.from_sequence(save_queue, npartitions=num_workers)
        save_bag.map(lambda x: x[0].save(x[1])).compute(
            scheduler=worker_type, num_workers=num_workers
        )

    @classmethod
    def load(
        cls,
        save_dir_path: str,
        worker_type: Literal["threads", "processes", "synchronous"] = "threads",
        num_workers: int | None = None,
    ) -> Self:

        metadata_save_path = os.path.join(save_dir_path, "metadata.json")
        if os.path.exists(metadata_save_path):
            with open(metadata_save_path) as f:
                metadata: dict = json.load(f)

        queue_name = metadata.get("queue_name", None)
        file_paths = metadata.get("file_paths", None)
        exp_names = metadata.get("exp_names", None)
        if exp_names is None:
            exp_data_dir = os.path.join(save_dir_path, "exp_data")
            if os.path.exists(exp_data_dir):
                exp_names = [
                    d \
                        for d in os.listdir(exp_data_dir) \
                            if os.path.isdir(os.path.join(exp_data_dir, d)) and d != "consensus_map"
                ]
        if exp_names is not None:

            exp_names = pl.Series(exp_names)

            consensus_map_save_path = os.path.join(exp_data_dir, "consensus_map")
            if os.path.exists(consensus_map_save_path):
                consensus_map = ConsensusMap.load(consensus_map_save_path)
            else:
                consensus_map = None

            spectrum_maps = []
            xic_maps = []
            feature_maps = []
            for exp_name in exp_names:
                exp_dir = os.path.join(exp_data_dir, exp_name)
                if os.path.exists(exp_dir):
                    if os.path.exists(os.path.join(exp_dir, "spectrum_map")):
                        spectrum_maps.append(os.path.join(exp_dir, "spectrum_map"))
                    else:
                        spectrum_maps.append(None)
                    if os.path.exists(os.path.join(exp_dir, "xic_map")):
                        xic_maps.append(os.path.join(exp_dir, "xic_map"))
                    else:
                        xic_maps.append(None)
                    if os.path.exists(os.path.join(exp_dir, "feature_map")):
                        feature_maps.append(os.path.join(exp_dir, "feature_map"))
                    else:
                        feature_maps.append(None)
            spectrum_maps = db.from_sequence(spectrum_maps, npartitions=num_workers).map(
                lambda x: SpectrumMap.load(x) if x is not None else None
            )
            xic_maps = db.from_sequence(xic_maps, npartitions=num_workers).map(
                lambda x: XICMap.load(x) if x is not None else None
            )
            feature_maps = db.from_sequence(feature_maps, npartitions=num_workers).map(
                lambda x: FeatureMap.load(x) if x is not None else None
            )
            spectrum_maps, xic_maps, feature_maps = dask.compute(
                spectrum_maps, xic_maps, feature_maps,
                scheduler=worker_type, num_workers=num_workers
            )
        else:
            spectrum_maps = None
            xic_maps = None
            feature_maps = None
            consensus_map = None

        return cls(
            queue_name=queue_name,
            file_paths=file_paths,
            exp_names=exp_names,
            spectrum_maps=spectrum_maps,
            xic_maps=xic_maps,
            feature_maps=feature_maps,
            consensus_map=consensus_map
        )


class MetaMSExperimentDataQueue(MetaMSDataWrapper):

    queue_name: str = Field(
        ...,
        description="队列名称"
    )
    file_paths: list[str] = Field(
        ...,
        description="原始文件路径列表"
    )
    exp_names: pl.Series = Field(
        ...,
        description="实验名称列表"
    )
    spectrum_maps: list[SpectrumMap] = Field(
        ...,
        description="谱图数据列表"
    )
    xic_maps: list[XICMap] = Field(
        ...,
        description="XIC数据列表"
    )
    feature_maps: list[FeatureMap] = Field(
        ...,
        description="特征图数据列表"
    )
