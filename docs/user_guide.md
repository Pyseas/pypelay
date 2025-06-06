# User guide

## Getting started
If you have followed the installation instructions then you have VS Code
open in a new folder, with a virtual environment into which you've
installed the *pypelay* package.

Copy and paste the below code block into a new file called *main.py*.

All of the commands in the code block are commented out. To run a command
you first uncomment it, run the Python script, then re-comment it.

    Uncomment line(s) -> Run script -> Comment line(s)

Work through the file running each command in sequence. The purpose of each
command is described in the sections below. 

<!-- ``` py linenums="1" -->

``` { .py .copy }
import pypelay
from pathlib import Path

PATH = Path('.')


def main():

    # ------------------------------------------------------------------
    # STEP 1 - New project setup
    # pypelay.fetch_files()

    # pypelay.list_raos('S1200')

    # vessel = pypelay.Vessel('S1200', vessel_type='vt S1200',
    #                         draft_name='Draft_7.4m')

    # ------------------------------------------------------------------
    # STEP 2 - Stinger configuration
    # pypelay.select_radius(vessel, num_section=3, water_depth=180,
    #                       tip_clearance=0.3, lcc_target=0.3)

    # pypelay.set_radius(vessel, num_section=3, radius=120,
    #                    water_depth=200, tip_clearance=0.3,
    #                    outpath=PATH / f'R120.dat')

    # pypelay.adjust_top_tension(inpath=PATH / 'R120.dat',
    #                            outpath=PATH / 'R120b.dat',
    #                            tension=30 * 9.81)

    # datpaths = [PATH / f'R120.dat', PATH / f'R120b.dat']
    # pypelay.static_summary(PATH / 'static_summary.xlsx', datpaths)

    # pypelay.write_dxf(PATH / 'R120b.dat')

    # ------------------------------------------------------------------
    # STEP 3 - Wave search
    # pypelay.wave_search(ncpu=8)

    # ------------------------------------------------------------------
    # STEP 4 - Dynamic simulations and post-processing

    # Create sims spreadsheet
    # pypelay.make_sims(PATH / 'R120_dyn.dat')

    # Run dynamic analysis
    # pypelay.run_sims(ncpu=8)

    # Post-process
    # pypelay.postprocess(PATH / 'summary.xlsx')


if __name__ == "__main__":
    main()

```

### Fetch input files
*pypelay* requires several input files containing pipe data, environment and
configuration options. Files can be manually copied across from another project,
or they can be obtained with [fetch_files][pypelay.fetch_files]:

```python
pypelay.fetch_files()
```

This will copy the below files into your project folder. Update the 
*pipe.dat* and *options.xlsx* files with your project data.
Instructions for populating the *environment.xlsx* spreadsheet are
given [below](#specify-environment). This can be done any time before
running dynamic analysis.

| File              |  Format   | Description               |
| --------------    | -------   | ------------------------- |
| pipe.dat          |  Orcaflex | Orcaflex dat file containing the pipe linetype   |
| options.xlsx      |  Excel    | Options for pipe segmentation and deadband  |
| environment.xlsx  |  Excel    | Wave and current data  |

### Select vessel

Display the list of available vessels and RAOs using
[list_raos][pypelay.list_raos]:

```python
pypelay.list_raos()
```

Select the vessel, vessel type and draft from the RAO list, and 
create the vessel. 

```python
vessel = pypelay.Vessel('S1200', 'vt S1200', 'Draft_7.4m')
```

Note that this line should remain uncommented since
it is required as an input to the *select_radius* and *set_radius*
commands.

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

Get the static results for any dat file, or compare multiple configurations
using [static_summary][pypelay.static_summary]. This will create a results
spreadsheet with one column for each file in *datpaths*.

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
Wave and current data required for dynamic analysis are specified in the
spreadsheet *environment.xlsx*. Inputs are described in the below
sections.

Populate the *wave_search* sheet and run the *wave_search* function. This will
generate a list of design wave windows, including seed number and time origin,
on the *waves* sheet.

```python
pypelay.wave_search(ncpu=8)
```

### Sheet: *hs_dirn*
Dynamic analysis will be run for each Hs and Direction combination.
For example 2x Hs and 3x Directions will create 6x combinations.

| Column        | Description               |
| ------------- | ------------------------- |
| hs            | List of Hs values (m)     |
| dirn          | List of directions (deg)  |
| dirn_name     | Name used for grouping results, e.g. 60, 90 and 120 can be named 'beam' |

### Sheet: *current*
Dynamic analysis is run for each direction specified in the direction column.
A different profile can be specified for each direction, or multiple profiles
can be run for a single direction. Current profiles are defined starting at column C.

| Column        | Description               |
| ------------- | ------------------------- |
| overall dirn      | Current direction     |
| overall profile   | Current profile number   |
| profile 1 depth   | Current profile 1 depth  |
| profile 1 speed   | Current profile 1 speed  |

### Sheet: *wave_search*
Input data required by the [wave_search][pypelay.wave_search] function:

| Column        |  Units  | Description               |
| ------------- |---------| ------------------------- |
| tp            |    s    | Wave peak period         |
| gamma         |    -    | JONSWAP peakedness parameter |
| hmax_factor   |    -    | Hmax = hmax_factor * Hs   |
| h_tol         |    m    | Hmax tolerance          |
| thmax_target  |    s    | Target period of Hmax    |
| t_tol         |    s    | thmax tolerance         |
| before        |    s    | Number of seconds in dynamic simulation before peak wave occurs |
| after         |    s    | Number of seconds in dynamic simulation after peak wave occurs |
| numseed       |    -    | Number of wave seeds to find |

### Sheet: *waves*
This sheet is automatically created when the user runs the
[wave_search][pypelay.wave_search] function.

## Dynamic analysis
Create the list of simulations using [make_sims][pypelay.make_sims]:

```python
pypelay.make_sims(PATH / 'R120_dyn.dat')
```

This will create the spreadsheet *sims.xlsx*. Open the spreadsheet and confirm
the list of simulations is as expected.

Run the simulations using [run_sims][pypelay.run_sims]:

```python
pypelay.run_sims(ncpu=8)
```

This can take several hours (or more) depending on the number of simulations.
On completion, all results are written to the spreadsheet *results.xlsx*.

## Post-processing
Post-process the results using [postprocess][pypelay.postprocess]:

```python
pypelay.postprocess(PATH / 'dyn_summary.xlsx')
```

## Concrete weight coating (CWC)
If the pipe has a CWC then the linetype category must be *General*, and some additional data must be specified using the linetype Tags.
The following tags must be present:

| Tag name        | Description               |
| ----------------| ------------------------- |
| cwc thickness (m)    | CWC thickness in m   |
| cwc density (t/m^3)  | CWC density in t/m^3 |
| lining thickness (m)    | Lining thickness in m   |
| lining density (t/m^3)  | Lining density in t/m^3 |
| snif  | Strain concentration factor |
| stress-strain table  | Name of stress-strain relationship |

The stress-strain table is a variable data source of type *Stress-strain relationship*. It must be a *Ramberg-Osgood curve* not a *Stress-strain table*.
