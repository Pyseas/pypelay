# User guide
If you have followed the installation instructions then you have VS Code
open in a workspace folder with a virtual environment, into which you've
installed the *pypelay* package.

Run the pipelay analysis by following each of the below sections in sequence.

## Getting started
Create a new python file, e.g. *main.py*.

At the top of the file, import the *pypelay* and *pathlib*
packages (*pathlib* is used to specify file system paths):

```python
import pypelay
from pathlib import Path

PATH = Path('.')
```

Next specify the vessel and its draft:

```python
vessel = pypelay.Vessel('S1200', draft=7400)
```

## Input files
*pypelay* requires several input files containing pipe data, environment and
configuration options. Copies of the input files are obtained with:

```python
pypelay.fetch_inputs()
```

This will copy the below files into your workspace folder. Update the files with
your project data.

| File             |  Format | Description               |
| --------------   | ------- | ------------------------- |
| pipe.dat         |  Orcaflex | Orcaflex dat file containing the pipe linetype   |
| [options.xlsx](spreadsheets.md#optionsxlsx) |  Excel | Options for pipe segmentation and deadband  |
| [environment.xlsx](spreadsheets.md#environmentxlsx) |  Excel | Wave and current data  |

## Select stinger radius
The fasest way to obtain the optimum stinger radius is using
[select_radius][pypelay.select_radius]:

```python
pypelay.select_radius(vessel, num_section=3, water_depth=180,
                      tip_clearance=0.3, lcc_target=0.3)
```

This will create 2 new Orcaflex dat files (one with pivoting rollers,
one with fixed rollers) with the automatically selected stinger radius.

To obtain a stinger configuration with a specific radius use
[set_radius][pypelay.set_radius]:

```python
pypelay.set_radius(vessel, num_section=3, radius=120, water_depth=180,
                   tip_clearance=0.3, outpath=PATH / 'R120.dat')
```

After a stinger configuration is created the top tension can be adjusted
using [adjust_top_tension][pypelay.adjust_top_tension]. Use this function
to set top tension to the closet 5t increment.

```python
pypelay.adjust_top_tension(PATH / 'R120.dat', PATH / 'R120b.dat',
                           tension=25*9.81)
```

Compare multiple configurations using [static_summary][pypelay.static_summary].
This will create a [static results](spreadsheets.md#static_summaryxlsx) spreadsheet.

```python
datpaths = [PATH / f'R120.dat', PATH / f'R120b.dat']
pypelay.static_summary(PATH / 'static_results.xlsx', datpaths)
```

## Create drawings
Drawings can be created at any time using [write_dxf][pypelay.write_dxf].
Two dxf files are created (one general arrangement, one roller details)
based on the input dat file containing the stinger configuration:

```python
pypelay.write_dxf(PATH / 'R120.dat')
```

## Specify environment


## Dynamic analysis


## Post-processing


