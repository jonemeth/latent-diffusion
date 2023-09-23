import importlib
import sys
from typing import Any, List, Optional, Type

import yaml


KEY_MODULES = "MODULES"
KEY_CLASS = "CLASS"
KEY_REF = "REF"
KEY_IMPORT = "IMPORT"


def find_class(modules: List[str], class_name: str) -> Type:
    class_modules = class_name.split('.')
    if len(class_modules) > 1:
        class_name = class_modules[-1]
        class_modules = ".".join(class_modules[:-1])
    else:
        class_modules = ""

    for module_name in modules:
        if len(module_name) > 0 and len(class_modules) > 0:
            full_module_name = module_name + "." + class_modules
        else:
            full_module_name = module_name + class_modules
        module = sys.modules.get(full_module_name)
        if module is None:
            module = importlib.import_module(full_module_name)
        if hasattr(module, class_name):
            return getattr(module, class_name)

    raise RuntimeError(f"Cannot find class {class_name}")


def process(cfg: Any, modules: List[str], parent_path: List[Any],
            keys: Optional[List[str]] = None) -> Any:
    if isinstance(cfg, dict):
        if KEY_IMPORT in cfg:
            assert 1 == len(cfg)
            return load_config(cfg[KEY_IMPORT])

        if KEY_REF in cfg:
            assert 1 == len(cfg)
            ref_name = cfg[KEY_REF]
            for ix, parent in reversed(list(enumerate(parent_path[:-1]))):
                if ref_name in parent:
                    return process(parent[ref_name], modules, parent_path[:ix])
            raise RuntimeError(f"Could not resolve reference {ref_name}!")

        class_name = cfg[KEY_CLASS] if KEY_CLASS in cfg.keys() else None
        if class_name is not None:
            del cfg[KEY_CLASS]

        # Note: Can't use list comprehension: objects can refer to already resolved objects.
        for k, v in cfg.items():
            if (not keys) or (k in keys):
                cfg[k] = process(v, modules, parent_path + [cfg])

        if class_name is not None:
            class_type = find_class(modules, class_name)
            return class_type(**cfg)

        return cfg

    if isinstance(cfg, list):
        return [process(v, modules, parent_path + [cfg]) for v in cfg]

    return cfg


def extract_modules(config: dict) -> List[str]:
    if KEY_MODULES not in config.keys():
        return []

    modules = config[KEY_MODULES]
    del config[KEY_MODULES]

    return modules


def load_config(yaml_file: str, keys: Optional[List[str]] = None) -> Any:
    with open(yaml_file, 'rt', encoding="utf-8") as file:
        config = yaml.safe_load(file)

    modules = extract_modules(config)
    config = process(config, modules, [], keys)

    return config
