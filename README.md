# New vessel setup
New vessel stinger setup functions are in prelim.py

``` py
    vessel = pypelay.Vessel('S1200', draft=7400)
    pypelay.valid_configs_to_df(vessel)
    pypelay.solve_configs()
    pypelay.combine_configs()
    pypelay.sort_configs()
    pypelay.plot_configs(num_section=3)
```

## valid_configs_to_df
- Calculate pipe paths, based on stinger radius, straight and transition lengths.
- For each pipe path find valid stinger settings, for 1-2-3 stinger sections (section angles, roller heights)
- Total # configs ~18400

## solve_configs
Solve all 18400 configs in parallel, using input water depth and stinger tip clearance. Iterate bollard pull to get target tip clearance.

Water depth comes from a made up formula, best fit to some early estimates of water depth vs stinger radius:

    Stinger radius = 73 + 8000 / depth

Results:
- Overbend UC
- Stinger tip angle and depth
- Stinger draft

Results are saved to individual spreadsheets for each num_section and radius combination.

## combine_configs
Combine all solved config spreadsheets into a single spreadsheet *solved_configs.xlsx*

## sort_configs
For each *num_section* (1-2-3) and *radius*, sort by overbend UC, then extract top 10 results and save to *top10_configs.xlsx*.

## plot_configs
Plot top 5 (i.e. lowest overbend UC) configs for each radius.
Manually select best config for each radius so that tip angle vs radius is monotonically decreasing and input this data to *preferred.xlsx*.
Preferred line will appear on scatter plot.
