
import OrcFxAPI as ofx
import math
import numpy as np
import pandas as pd
from copy import deepcopy
from multiprocessing import Pool
from dataclasses import dataclass, fields, asdict, replace
import sys
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
plt.style.use('ggplot')

PATH = Path('.')

__all__ = ['wave_search']


@dataclass
class Sim:
    tp: float
    gamma: float
    hmax_factor: float
    h_tol: float
    thmax_target: float
    t_tol: float
    before: float
    after: float
    numseed: int
    seed: int = 0
    t_origin: float = 0
    hmax: float = 0
    thmax: float = 0
    cross: str = 'x'

    def __str__(self):
        formatted = []
        for k, v in asdict(self).items():
            if k in ['seed', 'numseed', 'cross']:
                formatted.append(f'{v}')
            if k in ['before', 'after', 't_origin']:
                formatted.append(f'{v:.1f}')
            two_dp = ['tp', 'gamma', 'hmax_factor', 'h_tol',
                      'thmax_target', 't_tol', 'hmax', 'thmax']
            if k in two_dp:
                formatted.append(f'{v:.2f}')

        return '\t'.join(formatted)

    @property
    def header(self):
        return '\t'.join(x.name for x in fields(self))


def combine_results(xlpath: Path) -> None:

    txtpaths = (PATH / 'waves').glob('*.txt')

    dfs = []
    for pth in txtpaths:
        dfs.append(pd.read_csv(pth, delimiter='\t'))

    df = pd.concat(dfs)

    with pd.ExcelWriter(xlpath, engine='openpyxl', mode='a',
                        if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name='waves', index=False)


def interp(t: list[float], x: list[float]) -> float:
    # Returns zero-crossing time
    t0, t1 = t
    x0, x1 = x
    tc = t0 + (t1 - t0) * (-x0 / (x1 - x0))

    return tc


def zca(t, x, tol):

    zcind = np.where(np.diff(np.sign(x)))[0]

    # Remove peaks / troughs with height less than tol
    nzc = len(zcind)
    zcnew = []
    skip = False
    for izc in range(nzc - 1):
        if skip:
            skip = False
            continue
        ifrom = zcind[izc]
        ito = zcind[izc + 1]
        ht = abs(x[ifrom:ito]).max()
        if ht < tol:
            skip = True
        else:
            zcnew.append(zcind[izc])

    zcind = np.array(zcnew)
    zcind += 1

    upcross = None
    dncross = None

    for iupdn in range(2):

        cycles = []
        ncycle = math.floor(len(zcind) / 2)

        for i in range(ncycle - 1):

            ifrom = zcind[i * 2] - 1
            ito = zcind[i * 2 + 2] + 1

            tc0 = interp(t[ifrom:ifrom + 2], x[ifrom:ifrom + 2])
            tc1 = interp(t[ito - 2:ito], x[ito - 2:ito])

            sig = x[ifrom:ito]
            sig[0] = 0.0
            sig[-1] = 0.0

            tsig = t[ifrom:ito]
            tsig[0] = tc0
            tsig[-1] = tc1

            period = tc1 - tc0

            cycles.append([t[ifrom], period, sig.max(), sig.min()])

            if x[zcind[0] + 1] - x[zcind[0]] > 0:
                upcross = np.array(deepcopy(cycles))
            else:
                dncross = np.array(deepcopy(cycles))

        zcind = zcind[1:]

    return upcross, dncross


def waveheight(t: np.ndarray, elev: np.ndarray, tol: float):

    upcross, dncross = zca(t, elev, tol)

    if upcross is None or dncross is None:
        return [0, [0, 0, 0, 0], 'problem']

    upheights = abs(upcross[:, 2] - upcross[:, 3])
    upmax = upheights.max()
    upind = np.argmax(upheights)

    dnheights = abs(dncross[:, 2] - dncross[:, 3])
    dnmax = dnheights.max()
    dnind = np.argmax(dnheights)

    if upmax > dnmax:
        result = [upmax, upcross[upind, :], 'upcross']
    else:
        result = [dnmax, dncross[dnind, :], 'dncross']

    return result


def wavescreen(sim: Sim):

    model = ofx.Model()

    model.general.StageDuration[1] = 20 * 60

    env = model.environment
    env.SelectedWaveTrain = 'Wave1'
    env.WaveType = 'JONSWAP'
    env.UserSpecifiedRandomWaveSeeds = 'Yes'
    env.WaveHs = 1.0
    env.WaveGamma = sim.gamma
    env.WaveTp = sim.tp
    env.WaveDirection = 0.0

    results = []
    foundcount = 0

    for iseed in range(500):

        sim2 = replace(sim)

        seed = int((np.random.rand() - 0.5) * 1E9)
        env.WaveSeed = seed

        model.RunSimulation()

        # print(sim.tp, seed)
        # sys.stdout.flush()

        t = model.SampleTimes(1)
        elev = env.TimeHistory(
            'Elevation', 1, ofx.oeEnvironment((0.0, 0.0, 0.0)))

        hmax, param, updn = waveheight(t, elev, 0.01)
        t0, thmax, sigmax, sigmin = param

        cond1 = abs(hmax - sim2.hmax_factor) < sim2.h_tol
        cond2 = abs(thmax - sim2.thmax_target) < sim2.t_tol
        cond3 = t0 - sim.before - 8.0 > 0.0
        cond4 = t0 + sim.after < 1200.0

        if all([cond1, cond2, cond3, cond4]):
            foundcount += 1
            # No need to check other waveheights in window since whole 20 min
            # time series has already been checked
            t1 = t0 + thmax

            ind = np.where(np.all([
                [t >= t0 - sim2.before], [t <= t1 + sim2.after]],
                axis=0)[0])[0]

            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.plot(t[ind], elev[ind], color='royalblue')
            ax.set_xlabel('Simulation time (s)')
            ax.set_ylabel('Wave elevation (m)')
            title = f'Tp={sim.tp}s, seed={seed}, '
            title += f'Hmax={hmax:.2f}m, THmax={thmax:.2f}s'
            ax.set_title(title)

            figpath = PATH / 'waves' / f'{sim2.tp:02d}s_{seed}.pdf'
            plt.savefig(figpath)

            sim2.seed = seed
            sim2.t_origin = (t0 - sim.before) * -1
            sim2.hmax = hmax
            sim2.thmax = thmax
            sim2.cross = updn
            results.append(sim2)
            print(f'Tp {sim.tp}s {foundcount} seeds found')
            sys.stdout.flush()

        if foundcount == sim2.numseed:
            break

    # model.SaveData('sims\\LC_{0:05d}.dat'.format(lc))

    outstr = sim.header + '\n'
    for sim in results:
        outstr += str(sim) + '\n'

    with open(PATH / 'waves' / f'{sim.tp:03d}s.txt', 'w') as f:
        f.write(outstr)


def make_folders(folder_names: list[str]) -> bool:
    # Create new folder if folder doesn't exist
    # Prompt user to delete contents if folder does exist
    success = True
    for name in folder_names:
        fpath = PATH / name
        if fpath.exists():
            response = input(f'{name} folder already exists, delete contents (y/n)? : ')
            if response.lower() == 'y':
                files = fpath.glob('*.*')
                for file in files:
                    file.unlink()
            else:
                success = False
                break
        else:
            fpath.mkdir()

    return success


def wave_search(xlpath: Path, ncpu: int=8) -> None:
    """Identifies design wave windows, based on inputs in spreadsheet **environment.xlsx**
    For explanation of spreadsheet inputs refer to **Spreadsheets.md** help section

    Args:
        xlpath (Path): Path to *environment.xlsx* spreadsheet.
        ncpu (int): Number of CPUs to use.

    """

    df = pd.read_excel(xlpath, sheet_name='wave_search')

    sims = []
    for dic in df.to_dict('records'):
        sims.append(Sim(**dic))

    nsim = len(sims)

    print('Starting {0} sims with {1} processors'.format(nsim, ncpu))

    success = make_folders(['waves'])
    if not success:
        return

    with Pool(ncpu) as p:
        p.map(wavescreen, sims)

    # Read results files and write to spreadsheet
    combine_results(xlpath)


def main():
    pass


if __name__ == '__main__':
    main()
