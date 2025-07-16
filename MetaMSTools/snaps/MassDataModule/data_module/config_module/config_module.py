from typing import ClassVar

import toml
from pydantic import BaseModel, Field
from typing_extensions import Self


class TomlConfig(BaseModel):

    model_config = {"arbitrary_types_allowed": True}

    def to_dict(self):
        return self.model_dump()

    @classmethod
    def from_dict(cls, params: dict) -> Self:
        input_params = {}
        for param_name, param_meta in cls.model_fields.items():
            if hasattr(param_meta.annotation,"from_dict"):
                input_params[param_name] = param_meta.annotation.from_dict(params[param_name])
            else:
                input_params[param_name] = params[param_name]
        return cls(**input_params)

    def to_toml_string(self) -> str:
        return toml.dumps(self.model_dump())

    @classmethod
    def from_toml_string(cls, toml_string: str) -> Self:
        return cls.from_dict(toml.loads(toml_string))

    def to_toml(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            toml.dump(self.model_dump(), f)

    @classmethod
    def from_toml(cls, path: str) -> Self:
        return cls.from_dict(toml.load(path))

    def update(
        self,
        **kwargs,
    ):
        for key, value in kwargs.items():
            if key in type(self).model_fields:
                if isinstance(value, type(self).model_fields[key].annotation):
                    setattr(self, key, value)

    def get_runtime_config(
        self,
        **kwargs,
    ) -> Self:
        runtime_config = self.model_copy(update=kwargs)
        return runtime_config

class ConvertMethodConfig(TomlConfig):

    configs_type: ClassVar[dict[str, TomlConfig]] = {}

    method_name: str
    configs: dict[str, TomlConfig] = Field(default={})

    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        for config_name, config_type in self.configs_type.items():
            if config_name not in self.configs:
                self.configs[config_name] = config_type()
            if not isinstance(self.configs[config_name], config_type):
                if isinstance(self.configs[config_name], dict):
                    self.configs[config_name] = config_type.from_dict(self.configs[config_name])
                else:
                    self.configs[config_name] = config_type()

    @classmethod
    def from_dict(cls, params: dict) -> Self:
        input_params = {"method_name": params["method_name"]}
        configs_params = params['configs']
        for config_name, config_type in cls.configs_type.items():
            config_name: str
            config_type: TomlConfig
            if config_name in configs_params:
                input_params[config_name] = config_type.from_dict(configs_params[config_name])
            else:
                input_params[config_name] = config_type()
        for meta_name, meta_info in cls.model_fields.items():
            if meta_name not in input_params:
                if meta_name in params:
                    if hasattr(meta_info.annotation, "from_dict"):
                        input_params[meta_name] = meta_info.annotation.from_dict(params[meta_name])
                    else:
                        input_params[meta_name] = params[meta_name]
        return cls(**input_params)

    @property
    def config(self) -> TomlConfig:
        return self.configs[self.method_name]

    @config.setter
    def config(self, value: TomlConfig):
        self.configs[self.method_name] = value

    def __getitem__(self, key: str):
        if key == "method_name":
            return self.method_name
        elif key == "config":
            return self.config
        else:
            return self.configs[key]

    def __setitem__(self, key: str, value: TomlConfig | str):
        if key == "method_name":
            self.method_name = value
        elif key == "config":
            self.configs[key] = value
        else:
            self.configs[key] = value
