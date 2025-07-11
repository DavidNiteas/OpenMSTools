from __future__ import annotations

import copy
import json
import os
import pickle
from typing import Any

import pandas as pd
import polars as pl
import rtree
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import create_engine, text


class BaseMap(BaseModel):

    model_config = ConfigDict({"arbitrary_types_allowed": True})

    exp_name: str = Field(
        ...,
        data_type="metadata",
        save_mode="json",
        description="实验名称"
    )
    metadata: dict = Field(
        default={},
        data_type="metadata",
        save_mode="json",
        description="数据的metadata信息。"
    )

    def __getstate__(self):
        state = copy.deepcopy(super().__getstate__())
        for k,v in state['__dict__'].items():
            if isinstance(v, pl.DataFrame):
                state['__dict__'][k] = v.to_pandas()
        return state

    def __setstate__(self, state):
        init_func = []
        for k,f in self.model_fields.items():
            if f.annotation == pl.DataFrame or f.annotation == pl.DataFrame | None:
                if isinstance(state['__dict__'][k], pd.DataFrame):
                    state['__dict__'][k] = pl.from_pandas(state['__dict__'][k])
            elif f.annotation == rtree.index.Index | None:
                if isinstance(state['__dict__'][k], rtree.index.Index):
                    if "init_func" in f.json_schema_extra:
                        init_func.append(f.json_schema_extra["init_func"])
            elif f.annotation == rtree.index.Index:
                if "init_func" in f.json_schema_extra:
                    init_func.append(f.json_schema_extra["init_func"])
        super().__setstate__(state)
        for func_name in init_func:
            getattr(self, func_name)()

    def save(self, save_dir_path: str):

        if not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path)

        metadata_path = os.path.join(save_dir_path, "metadata.json")
        index_dir_path = os.path.join(save_dir_path, "index")
        if not os.path.exists(index_dir_path):
            os.makedirs(index_dir_path)
        data_dir_path = os.path.join(save_dir_path, "data")
        if not os.path.exists(data_dir_path):
            os.makedirs(data_dir_path)
        sqlite_db_path = os.path.join(data_dir_path, "data.sqlite")
        engine = create_engine(f"sqlite:///{sqlite_db_path}")

        metadata_to_save = {"module_type": self.__class__.__name__}
        for k,f in self.model_fields.items():
            if f.json_schema_extra['data_type'] == 'metadata':
                metadata_to_save[k] = getattr(self, k)
            elif f.json_schema_extra['data_type'] == 'index':
                if isinstance(getattr(self, k), rtree.index.Index):
                    rtree_save_path = os.path.join(index_dir_path, k)
                    if os.path.exists(rtree_save_path + ".dat"):
                        os.remove(rtree_save_path + ".dat")
                    if os.path.exists(rtree_save_path + ".idx"):
                        os.remove(rtree_save_path + ".idx")
                    tree:rtree.index.Index = getattr(self, f.json_schema_extra['build_func'])(rtree_save_path)
                    tree.close()
                elif isinstance(getattr(self, k), pd.Index):
                    index_save_path = os.path.join(index_dir_path, k+".csv")
                    pd.Series(getattr(self, k)).to_csv(index_save_path, header=False)
                else:
                    other_index_save_path = os.path.join(index_dir_path, k+".pkl")
                    with open(other_index_save_path, 'wb') as f:
                        pickle.dump(getattr(self, k), f)
            elif f.json_schema_extra['data_type'] == 'data':
                if f.json_schema_extra['save_mode'] == 'sqlite':
                    data = getattr(self, k)
                    if isinstance(data, pl.DataFrame):
                        with engine.connect() as conn:
                            data.write_database(table_name=k, connection=conn, if_table_exists="replace")
                    elif isinstance(data, pd.DataFrame):
                        with engine.connect() as conn:
                            data.to_sql(k, conn, if_exists="replace")
                    else:
                        raise ValueError(f"Unsupported data type to save as sqlite: {type(data)}")
                else:
                    other_data_save_path = os.path.join(data_dir_path, k+".pkl")
                    with open(other_data_save_path, 'wb') as f:
                        pickle.dump(getattr(self, k), f)

        with open(metadata_path, 'w') as f:
            json.dump(metadata_to_save, f)

        engine.dispose()

    @classmethod
    def _base_load(cls, save_dir_path: str) -> dict[str, Any]:

        data_dict = {}

        metadata_path = os.path.join(save_dir_path, "metadata.json")
        index_dir_path = os.path.join(save_dir_path, "index")
        data_dir_path = os.path.join(save_dir_path, "data")

        if not os.path.exists(metadata_path):
            raise ValueError(f"Metadata file not found in {save_dir_path}")
        with open(metadata_path) as f:
            metadata:dict = json.load(f)
            exp_name = metadata.pop('exp_name')
            metadata.pop('module_type')
        data_dict['exp_name'] = exp_name
        data_dict['metadata'] = metadata

        if os.path.exists(index_dir_path):
            for k,f in cls.model_fields.items():
                if f.json_schema_extra['data_type'] == 'index':
                    if f.json_schema_extra['save_mode'] == 'rtree':
                        rtree_save_path = os.path.join(index_dir_path, k)
                        if os.path.exists(rtree_save_path + ".dat") and os.path.exists(rtree_save_path + ".idx"):
                            data_dict[k] = rtree.index.Index(rtree_save_path)
                    elif f.annotation == pd.Index or f.annotation == pd.Index | None:
                        index_save_path = os.path.join(index_dir_path, k+".csv")
                        if os.path.exists(index_save_path):
                            data_dict[k] = pd.Index(pd.read_csv(index_save_path, header=None, index_col=0).iloc[:,0])
                    else:
                        other_index_save_path = os.path.join(index_dir_path, k+".pkl")
                        if os.path.exists(other_index_save_path):
                            with open(other_index_save_path, 'rb') as f:
                                data_dict[k] = pickle.load(f)

        if os.path.exists(data_dir_path):
            sqlite_db_path = os.path.join(data_dir_path, "data.sqlite")
            if os.path.exists(sqlite_db_path):
                engine = create_engine(f"sqlite:///{sqlite_db_path}")
                for k,f in cls.model_fields.items():
                    if f.json_schema_extra['data_type'] == 'data':
                        if f.json_schema_extra['save_mode'] == 'sqlite':
                            with engine.connect() as conn:
                                if conn.execute(text(
                                    f"SELECT name \
                                        FROM sqlite_master \
                                        WHERE type='table' \
                                        AND name='{k}'"
                                )).fetchone() is not None:
                                    if f.annotation == pl.DataFrame or f.annotation == pl.DataFrame | None:
                                        data_dict[k] = pl.read_database(query=f"SELECT * FROM {k}", connection=conn)
                                    elif f.annotation == pd.DataFrame or f.annotation == pd.DataFrame | None:
                                        data_dict[k] = pd.read_sql_query(f"SELECT * FROM {k}", conn)
                        else:
                            other_data_save_path = os.path.join(data_dir_path, k+".pkl")
                            if os.path.exists(other_data_save_path):
                                with open(other_data_save_path, 'rb') as f:
                                    data_dict[k] = pickle.load(f)
                engine.dispose()

        return data_dict

    @classmethod
    def load(cls, save_dir_path: str):

        return cls(**cls._base_load(save_dir_path))

