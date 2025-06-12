from pydantic import BaseModel, Field, ConfigDict
from abc import ABC,abstractmethod
import toml
import pyopenms as oms
import pandas as pd
from typing_extensions import Self
from typing import Type, Dict, Any, Union, Optional, ClassVar, List

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
    
class OpenMSParam(ABC):
    
    @abstractmethod
    def size(self) -> int:
        pass
    
    @abstractmethod
    def keys(self) -> List[bytes]:
        pass
    
    @abstractmethod
    def values(self) -> List[Any]:
        pass
    
    @abstractmethod
    def __getitem__(self, key: bytes) -> Any:
        pass
    
    @abstractmethod
    def descriptions(self) -> List[str]:
        pass
    
    @abstractmethod
    def getDescription(self,key:bytes) -> str:
        pass
    
class OpenMSMethod(ABC):
    
    @abstractmethod
    def getParameters(self) -> OpenMSParam:
        pass
    
    @abstractmethod
    def getDefaults(self) -> OpenMSParam:
        pass
    
class OpenMSMethodParam(ABC, TomlConfig):
    
    @abstractmethod
    def dump2openms(self) -> Dict[str, Any]:
        pass
    
class OpenMSMethodParamWrapper(OpenMSMethodParam):
    
    wrapper_name: ClassVar[str]
    
    def dump2openms(self) -> Dict[str, Any]:
        param_dict = {}
        for field_name in self.model_fields:
            field_data = getattr(self, field_name)
            if isinstance(field_data, OpenMSMethodParam):
                for openms_key, openms_value in field_data.dump2openms().items():
                    param_dict[f"{self.wrapper_name}:{openms_key}"] = openms_value
            else:
                param_dict[f"{self.wrapper_name}:{field_name}"] = field_data
        return param_dict

class OpenMSMethodConfig(ABC, TomlConfig):
    
    openms_method: ClassVar[Type[OpenMSMethod]]
    
    @property
    def param(self) -> OpenMSParam:
        param = oms.Param()
        for key in self.model_fields:
            value = getattr(self, key)
            if isinstance(value, OpenMSMethodParam):
                openms_dict = value.dump2openms()
                for openms_key, openms_value in openms_dict.items():
                    param.setValue(openms_key, openms_value)
            else:
                param.setValue(key, value)
        return param
    
    @property
    def param_info(self) -> pd.DataFrame:
        method = self.openms_method()
        default_param = method.getDefaults()
        default_config = self.__class__()
        infos = {"value":{},"default_value":{},"openms_default_value":{},"description":{}}
        names = default_param.keys()
        for name in names:
            infos["openms_default_value"][name.decode()] = default_param[name]
            infos["description"][name.decode()] = default_param.getDescription(name)
        for field_name in self.model_fields:
            field_data = getattr(self, field_name)
            if isinstance(field_data, OpenMSMethodParam):
                for name, value in field_data.dump2openms().items():
                    infos["value"][name] = value
            else:
                infos["value"][field_name] = field_data
        for field_name in default_config.model_fields:
            field_data = getattr(default_config, field_name)
            if isinstance(field_data, OpenMSMethodParam):
                for name, value in field_data.dump2openms().items():
                    infos["default_value"][name] = value
            else:
                infos["default_value"][field_name] = field_data
        infos = pd.DataFrame(infos)
        infos.index.name = "param_name"
        return infos
    
class ConvertMethodConfig(TomlConfig):
    
    configs_type: ClassVar[Dict[str, TomlConfig]] = {}
    
    method_name: str
    configs: Dict[str, TomlConfig] = Field(default={})
    
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        for config_name, config_type in self.configs_type.items():
            if config_name not in self.configs:
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

    def __setitem__(self, key: str, value: Union[TomlConfig, str]):
        if key == "method_name":
            self.method_name = value
        elif key == "config":
            self.configs[key] = value
        else:
            self.configs[key] = value
    
class MSToolConfig(TomlConfig):
    
    pass

class OpenMSDataWrapper(BaseModel):
    
    model_config = ConfigDict({"arbitrary_types_allowed": True})
    
    file_paths: Optional[List[str]] = None
    exp_names: Optional[List[str]] = None
    exps: Optional[List[oms.MSExperiment]] = None
    mass_traces: Optional[List[List[oms.MassTrace]]] = None
    features: Optional[List[oms.FeatureMap]] = None
    consensus_maps: Optional[oms.ConsensusMap] = None
    ref_feature: Optional[oms.FeatureMap] = None
    ref_exp: Optional[List[oms.MSExperiment]] = None
    ref_file_paths: Optional[List[str]] = None
    trafos: Optional[List[oms.TransformationDescription]] = None

class MSTool(ABC):
    
    config_type: Union[Type[MSToolConfig],Type[ConvertMethodConfig],Type[OpenMSMethodConfig],Type[TomlConfig]]
    
    def __init__(
        self,
        config: Union[MSToolConfig,ConvertMethodConfig,OpenMSMethodConfig,TomlConfig,None] = None,
    ):
        if not isinstance(config, self.config_type):
            self.config = self.config_type()
        else:
            self.config = config
    
    @abstractmethod
    def __call__(self, data: OpenMSDataWrapper) -> OpenMSDataWrapper:
        pass