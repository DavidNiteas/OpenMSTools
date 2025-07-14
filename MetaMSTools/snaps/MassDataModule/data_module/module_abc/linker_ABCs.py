from typing import ClassVar

from pydantic import BaseModel, ConfigDict

class BaseLinker(BaseModel):

    model_config = ConfigDict({"arbitrary_types_allowed": True})

    