import pickle
import io
import importlib
import sys 
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Mapping of old module names to new module names
MODULE_RENAME_MAP = {
    'module': 'diffu_models',
    'module.model': 'diffu_models.precond',
    'module.dit_models': 'diffu_models.dit_models',
    'module.model': 'diffu_models.precond',
    # add more as needed
}

class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in MODULE_RENAME_MAP:
            module = MODULE_RENAME_MAP[module]
        try:
            return super().find_class(module, name)
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(f"Could not find module '{module}'. You may need to update MODULE_RENAME_MAP.") from e

# Usage
def load_renamed_pickle(file_path):
    with open(file_path, 'rb') as f:
        return RenameUnpickler(f).load()

def safe_load_with_fallback(file_path):
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except ModuleNotFoundError:
        with open(file_path, 'rb') as f:
            return RenameUnpickler(f).load()