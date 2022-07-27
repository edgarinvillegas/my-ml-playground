# ML utils lib (WIP)

Collection of my utilities. This is WIP.
Current purpose is local development 

## Lib consumption (once distributed)
- `pip install my_ml_utils_dist`
- `from train_validate_package import train_validate` 

## Lib development

### 1. Install library to local environment (hot reload)
- Activate your preferred environment
- Go to this folder (where setup.py is)
- `pip install -e .` (setuptools develop mode, creates symlink)

### 2. Consume the lib
- Same steps as **Lib consumption**, but it will install from created symlink 
- If using jupyter, handy way to autoreload:
```
%load_ext autoreload
%autoreload 1
%aimport train_validate_package

from train_validate_package import train_validate
```
- Change the code and it will automatically reload on jupyter

## Resources
- [When to use pip install -e](https://stackoverflow.com/questions/42609943/what-is-the-use-case-for-pip-install-e)
- [Jupyter autoreload](https://stackoverflow.com/questions/49264194/import-py-file-in-another-directory-in-jupyter-notebook)

## TODO
- Remove `*package` and `*module` suffixes in file names
- Move reusable files to a lib folder. Separate sample notebooks to another dir
- Clean and publish to pypi