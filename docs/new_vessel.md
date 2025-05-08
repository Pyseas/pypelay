# New vessel setup
New vessel stinger setup functions are in prelim.py

``` py
    vessel = pypelay.Vessel('S1200', draft=7400)
    # radii = list(np.linspace(80, 250, 35) * 1000)
    # radii += list(np.linspace(260, 350, 10) * 1000)
    # radii += list(np.linspace(370, 550, 10) * 1000)
    pypelay.valid_configs_to_df(vessel, radii)
    # Move valid_configs spreadsheet into new *configs* folder
    pypelay.solve_configs(vessel, radii)
    pypelay.combine_configs()
    pypelay.sort_configs()
    pypelay.plot_configs(num_section=3, radii=radii)
```

## valid_configs_to_df
- Calculate pipe paths, based on stinger radius, straight and transition lengths.
- For each pipe path find valid stinger settings, for 1-2-3 stinger sections (section angles, roller heights)
- Total # configs can be very large 10000+

**IMPORTANT**
Manually add lc column to *valid_configs.xlsx*. Avoids headaches with following steps.

## solve_configs
Solve all configs in parallel, using input water depth and stinger tip clearance. Iterate bollard pull to get target tip clearance.

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
