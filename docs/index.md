# Introduction
pypelay is a Python library for S-lay stinger configuration and pipelay analysis.

## Installation
Use pip to install pypelay wheel
```bash
pip install pypelay-0.1.0-py3-none-any.whl
```

## Usage
```python
import pypelay

# Creates a new sacinp file using data from LiftModel spreadsheet
pypelay.make_new_model('LiftModel.xlsx', 'sacinp.base', 'sacinp.lift')

```
