
import numpy as np
import pandas as pd
from multiprocessing import Pool
import OrcFxAPI as ofx
from dataclasses import astuple
from pathlib import Path
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from .geom import calc_path_coords
from .stinger import (get_base_case, StingerSetupArgs, stinger_setup,
                      get_roller_heights, Vessel)

PATH = Path('.')

__all__ = ['valid_configs_to_df', 'solve_configs', 'combine_configs',
           'sort_configs', 'plot_configs', 'write_final_configs']


def write_final_configs(vessel: Vessel, inpath: Path) -> None:

    df = pd.read_excel(inpath)
    df = df[df['prefer'] == 1]
    df.reset_index(drop=True, inplace=True)
    nconfig = len(df)

    # random_configs = [4, 5, 38, 50, 62, 68, 70, 75, 75, 94, 104, 105, 118, 153, 165]
    # df = df.iloc[random_configs]

    df.drop(['prefer'], axis=1, inplace=True)

    icfg = 0
    roller_dicts = {}
    for row in df.itertuples():
        print(row.radius, row.num_section, f'{icfg + 1}/{nconfig}')
        icfg += 1
        # Create base case model
        model = get_base_case(vessel, row.radius, row.num_section)
        path_coords = calc_path_coords(model, row.straight, row.transition)
        rollers = get_roller_heights(model, path_coords, row.ang1, row.ang2)
        roller_dict = {}
        for roller in rollers:
            roller_dict[f'{roller.name} y'] = roller.y
            roller_dict[f'{roller.name} r3'] = roller.r3
            roller_dict[f'{roller.name} post_angle'] = roller.post_angle
            roller_dict[f'{roller.name} arc'] = roller.arc
            roller_dict[f'{roller.name} y_offset'] = roller.y_offset
        roller_dicts[row.lc] = roller_dict

    df2 = pd.DataFrame(roller_dicts).transpose()

    df = df.join(df2, on='lc', how='left')

    df.to_excel(PATH / 'configs' / f'{vessel.name}_configs.xlsx', index=False)

    return


def plot_configs(num_section: int, radii: list[float]) -> None:

    # Get preferred config for each radius
    prefer = pd.read_excel(PATH / 'configs' / 'preferred.xlsx')
    for i in [1, 2, 3]:
        if i != num_section:
            prefer.drop(i, axis=1, inplace=True)
    prefer.columns = ['radius', 'rank']
    ind = dict(zip(prefer['radius'], prefer['rank']))

    df = pd.read_excel(PATH / 'configs' / 'top10_configs.xlsx')

    to_plot = []
    prefer = []
    for rad in radii:
        df2 = df[(df['radius'] == rad) & (df['num_section'] == num_section)]
        if len(df2) == 0:
            continue
        row = int(ind[rad/1000])
        to_plot.append([rad, df2['tip_angle'].iloc[row]])
        prefer.append(df2['lc'].iloc[row])
    to_plot = np.asarray(to_plot)

    # Update prefer column to indicate preferred config for each radius
    df.loc[df['num_section'] == num_section, 'prefer'] = 0
    df.loc[df['lc'].isin(prefer), 'prefer'] = 1
    df.to_excel(PATH / 'configs' / 'top10_configs.xlsx', index=False)

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    gb = df[df['num_section'] == num_section].groupby('radius')
    for ipos in [6, 5, 4, 3, 2, 1, 0]:
        df2 = gb.nth(ipos).reset_index()
        ax.scatter(df2['radius'], df2['tip_angle'], label=ipos)

    ax.set_xlabel('Stinger radius (mm)')
    ax.set_ylabel('Stinger tip angle (deg)')

    ax.plot(to_plot[:, 0], to_plot[:, 1])
    plt.suptitle(f'{num_section} stinger sections')
    plt.legend()
    plt.show()


def sort_configs():

    df = pd.read_excel(PATH / 'configs' / 'solved_configs.xlsx')

    # Sort results by overbend UC and extract top 10 configs for each
    # num_section / radius combination
    dfs = []
    for num_section in [1, 2, 3]:
        df2 = df[df['num_section'] == num_section]
        df2 = df2.sort_values(['uc_ob']).groupby('radius').head(10)
        df2 = df2.reset_index(drop=True).sort_values(['radius', 'uc_ob'])
        dfs.append(df2)

    df = pd.concat(dfs)
    df['prefer'] = 0
    df.to_excel(PATH / 'configs' / 'top10_configs.xlsx', index=False)


def combine_configs():

    # Combine result spreadsheets into a single spreadsheet
    # xlpaths = []
    # for radius in np.linspace(80, 250, 35) * 1000:
    #     for num_section in [1, 2, 3]:
    #         xlname = f'configs_{num_section}_{radius/1000:.0f}.xlsx'
    #         xlpaths.append(PATH / 'configs' / xlname)

    xlpaths = (PATH / 'configs').glob('configs_*.xlsx')

    dfs = []
    for p in xlpaths:
        df = pd.read_excel(p)
        dfs.append(df)

    df = pd.concat(dfs)

    df.to_excel(PATH / 'configs' / 'solved_configs.xlsx', index=False)


def solve_configs(vessel: Vessel, radii: list[float]) -> None:

    df = pd.read_excel(PATH / 'configs' / 'valid_configs.xlsx')

    for radius in radii:
        for num_section in [1, 2, 3]:
            df2 = df[(df['radius'] == radius) &
                     (df['num_section'] == num_section)]
            # rerun = list(range(6585, 6600))
            # df2 = df[(df['lc'].isin(rerun))]

            model = get_base_case(vessel, radius, num_section)

            inpath = PATH / 'base case.dat'

            water_depth = int(8000 / (radius/1000 - 73))

            tip_clearance = 0.2

            sims = []
            # result = []
            for config in df2.to_dict(orient="records"):
                lc = int(config['lc'])
                outpath = PATH / 'sims' / f'LC_{lc:05d}.dat'
                # setup_args = StingerSetupArgs(
                #         inpath, outpath, config, water_depth, tip_clearance,
                #         delete_dat=True)
                # result.append(stinger_setup(setup_args))
                sims.append(
                    StingerSetupArgs(
                        inpath, outpath, config, water_depth, tip_clearance,
                        delete_dat=True))

            with Pool(8) as p:
                result = p.map(stinger_setup, sims, chunksize=1)

            # result is list of StingerSetupResults
            res_table = []
            for res in result:
                lc = int(res.outpath.stem.split('_')[1])
                res_table.append([lc] + list(astuple(res))[1:])

            cols = ['lc', 'top_tension', 'uc_ob', 'uc_sag', 'tip_depth', 'tip_angle', 'draft']
            res = pd.DataFrame(res_table, columns=cols)
            df2 = df2.merge(res)

            outpath = PATH / 'configs' / f'configs_{num_section}_{radius/1000:.0f}.xlsx'
            df2.to_excel(outpath, index=False)


def get_valid_configs(sim) -> list[list[float]]:
    ''' Loops through section angles to find valid stinger configs'''
    straight, transition, path_coords, section_angles = sim

    model = ofx.Model(PATH / 'base case.dat')

    results = []
    for ang1, ang2 in section_angles:
        rollers = get_roller_heights(model, path_coords, ang1, ang2)
        if rollers:
            results.append([straight, transition, ang1, ang2])

    return results


def valid_configs_to_df(vessel: Vessel, radii: list[float]) -> None:
    ''' Finds valid stinger configurations for all stinger radii
        (and num_section) and saves them to valid_configs.xlsx
        stinger_config =
            radius, num_section, straight, transition, ang1, ang2'''

    nconfig = len(radii) * 3

    icfg = 0
    configs = []
    for radius in radii:
        for num_section in [1, 2, 3]:
            print(radius, num_section, f'{icfg + 1}/{nconfig}')
            # Create base case model
            model = get_base_case(vessel, radius, num_section)

            stinger_ref = model['b6 stinger_ref']
            tensioner_x = float(stinger_ref.tags['tensioner_x'])

            section_angles = []
            a1 = np.linspace(0, 45, 91)
            a2 = [0, 10, 15, 20, 25, 30]
            if num_section == 1:
                a2 = [0]
            for ang1 in a1:
                for ang2 in a2:
                    section_angles.append([ang1, ang2])

            sims = []
            for straight in np.linspace(0, 16000, 9):
                maxtrans = tensioner_x - straight
                ntrans = round(maxtrans / 2000)
                dtrans = maxtrans / ntrans
                for transition in np.linspace(dtrans, maxtrans, ntrans):
                    path_coords = calc_path_coords(model, straight, transition)
                    sims.append([straight, transition, path_coords, section_angles])

            with Pool(8) as p:
                result = p.map(get_valid_configs, sims, chunksize=1)

            for res in result:
                for row in res:
                    configs.append([radius, num_section] + row)

            icfg += 1

    cols = ['radius', 'num_section', 'straight', 'transition', 'ang1', 'ang2']

    configs = pd.DataFrame(np.asarray(configs), columns=cols)

    configs.to_excel(PATH / 'valid_configs.xlsx', index=False)


def main():
    pass


if __name__ == "__main__":
    main()

