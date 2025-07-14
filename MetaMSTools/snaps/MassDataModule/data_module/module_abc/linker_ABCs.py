import os

import polars as pl
from pydantic import BaseModel, ConfigDict
from sqlalchemy import create_engine, inspect
from typing_extensions import Self


class BaseLinker(BaseModel):

    model_config = ConfigDict({"arbitrary_types_allowed": True})

    def save(self, save_path: str) -> None:

        engine = create_engine(f"sqlite:///{save_path}")

        for key, value in self:
            if isinstance(value, pl.DataFrame):
                with engine.begin() as conn:
                    value.write_database(key, conn, if_table_exists="replace")
            elif key == "exp_names":
                with engine.begin() as conn:
                    pl.DataFrame({key: value}).write_database(key, conn, if_table_exists="replace")
            elif key == "queue_name":
                with engine.begin() as conn:
                    pl.DataFrame({key: [value]}).write_database(key, conn, if_table_exists="replace")

        engine.dispose()

    @classmethod
    def load(cls, load_path: str) -> Self:
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"File {load_path} does not exist.")

        engine = create_engine(f"sqlite:///{load_path}")
        insp = inspect(engine)
        tables = insp.get_table_names()

        kwargs: dict = {}
        for tbl in tables:
            df = pl.read_database(f"SELECT * FROM {tbl}", engine)
            if tbl == "exp_names":
                kwargs[tbl] = pl.Series(df[tbl].to_list())
            elif tbl == "queue_name":
                kwargs[tbl] = df[tbl][0]
            else:
                kwargs[tbl] = df

        engine.dispose()

        return cls(**kwargs)
