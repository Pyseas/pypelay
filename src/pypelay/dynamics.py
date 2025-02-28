
from dataclasses import dataclass, fields
import OrcFxAPI as ofx
from pathlib import Path
from multiprocessing import Pool
import pandas as pd
import OrcFxAPI as ofx

PATH = Path('.')

__all__ = ['make_sims', 'run_sims', 'postprocess', 'make_dat']


@dataclass
class Sim:
    base_filename: str
    lc: int
    hs: float
    tp: float
    gamma: float
    dirn: float
    dirn_name: str
    seed: int
    t_origin: float
    duration: float
    cspd: float
    cdirn: float
    cprofile: int
    run_sim: bool = True
    save_sim: bool = False
    save_dat: bool = False

    # def __str__(self):
    #     mystr = f'{self.lc}\t{self.hs}\t{self.tp:.2f}\t{self.gamma:.2f}\t'
    #     mystr += f'{self.dirn:.1f}\t{self.seed:.0f}\t{self.t_origin:.1f}\t'
    #     return mystr

    @property
    def header(self):
        return [x.name for x in fields(self)]


def aggfuncs(df: pd.DataFrame) -> dict:

    lookup = {'static': 'max', 'max': 'max', 'min': 'min'}
    aggs = {}
    for col in df.columns:
        aggs[col] = lookup[col.split()[-1]]

    return aggs


def get_header() -> list[str]:
    # Results column names
    header = ['lc']
    for var in ['layback', 'susp_length', 'top_tension',
                'tdp_tension', 'tip_clearance']:
        for res in ['static', 'max', 'min']:
            header.append(f'{var} {res}')

    for var in ['barge_roller_load', 'stinger_roller_load',
                'f101a_overbend', 'f101a_tip', 'f101a_sag',
                'f101b_overbend', 'f101b_tip', 'f101b_sag']:
        for res in ['static', 'max']:
            header.append(f'{var} {res}')

    return header


def make_sims(inpath: Path, base_filename: str) -> None:

    df0 = pd.read_excel(inpath, sheet_name='waves')
    df1 = pd.read_excel(inpath, sheet_name='hs_dirn')
    df2 = pd.read_excel(inpath, sheet_name='current', header=[0, 1])

    df0['duration'] = df0['before'] + df0['after']
    drop_cols = [
        'hmax_factor', 'thmax_target', 'h_tol', 't_tol',
        'before', 'after', 'numseed', 'hmax', 'thmax', 'cross']
    df0.drop(drop_cols, axis=1, inplace=True)

    dfs = []
    for hs in df1['hs'].dropna():
        df1['hs'] = hs
        dfs.append(df1.copy())
    df1 = pd.concat(dfs, ignore_index=True)

    df0['join'] = 0
    df1['join'] = 0

    sims = df0.merge(df1, on='join', how='outer')
    sims['base_filename'] = base_filename
    sims.drop(['join'], axis=1, inplace=True)
    sims['lc'] = range(1, len(sims.index) + 1)
    cols = ['lc', 'base_filename', 'hs', 'tp', 'gamma', 'dirn',
            'dirn_name', 'seed', 't_origin', 'duration']
    sims = sims[cols]

    for var in ['cspd', 'cdirn', 'cprofile']:
        sims[var] = 0.0

    dfs = [sims.copy()]
    df3 = df2['overall'].dropna()
    if len(df3.index) > 0:
        # write current_profiles to csv
        df2.drop([('overall', 'dirn'), ('overall', 'profile')], axis=1, inplace=True)
        df2.to_csv(PATH / 'current_profiles.csv')
        for row in df3.itertuples():
            sims['cspd'] = 1.0
            sims['cdirn'] = row.dirn
            sims['cprofile'] = row.profile
            dfs.append(sims.copy())

    sims = pd.concat(dfs, ignore_index=True)
    sims['lc'] = range(1, len(sims.index) + 1)

    outpath = PATH / 'sims.xlsx'
    sims.to_excel(outpath, index=False)


def fix_rollers(datpath: Path) -> None:

    model = ofx.Model(datpath)

    model.CalculateStatics()

    all_names = [obj.Name for obj in model.objects]

    roller_names = [x[3:] for x in all_names if x[:5] in ['b6 BR', 'b6 SR']]

    # Determine if roller angle needs to be set to r3 based on roller load
    adjust_angle = {}
    for rname in roller_names:
        oroller = model[f'b6 {rname}']
        nsup = oroller.NumberOfSupports
        for isup in range(nsup):
            objx = ofx.oeSupport(isup + 1)
            sup_load = oroller.StaticResult('Support reaction force', objx)
            adjust_angle[rname] = False
            if sup_load < 0.1:
                adjust_angle[rname] = True
                break

    for rname in roller_names:
        oroller = model[f'b6 {rname}']
        ocn = model[f'cn {rname}']
        oroller.Connection = ocn.Connection
        if adjust_angle[rname]:
            r3 = float(oroller.tags['r3'])
            oroller.InitialRotation3 = r3
        model.DestroyObject(ocn)

    outpath = datpath.parent / f'{datpath.stem}_fixed.dat'
    model.SaveData(outpath)


def get_roller_loads(roller_names: list[str],
                     model: ofx.Model) -> tuple[float, float]:

    static = []
    dyn = []
    for rname in roller_names:
        oroller = model[f'b6 {rname}']
        var = 'Support reaction force'
        res0 = (oroller.StaticResult(var, ofx.oeSupport(1)) +
                oroller.StaticResult(var, ofx.oeSupport(2)))
        res1 = (oroller.TimeHistory(var, 1, ofx.oeSupport(1)) +
                oroller.TimeHistory(var, 1, ofx.oeSupport(2)))
        static.append(res0)
        dyn.append(res1.max())

    return max(static), max(dyn)


def make_dat(tp: list[float], dirn: list[float]) -> None:

    df = pd.read_excel(PATH / 'sims.xlsx')
    df = df[(df['tp'].isin(tp)) & (df['dirn'].isin(dirn))]

    outfolder = PATH / 'datfiles'
    if not outfolder.exists():
        outfolder.mkdir()

    for sim in df.itertuples():
        base_path = PATH / str(sim.base_filename)
        model = ofx.Model(base_path)

        ovessel = [obj for obj in model.objects if obj.typeName == 'Vessel'][0]
        vtype = model[ovessel.VesselType]
        env = model.environment
        env.SelectedWaveTrain = 'Wave1'
        env.WaveHs = sim.hs
        env.WaveGamma = sim.gamma
        env.WaveTp = sim.tp
        env.WaveDirection = sim.dirn
        env.WaveSeed = sim.seed
        env.WaveOriginX = ovessel.InitialX - vtype.Length / 2
        env.WaveTimeOrigin = sim.t_origin
        model.general.StageDuration[1] = sim.duration

        outpath = outfolder / f'LC_{sim.lc:05d}.dat'
        model.SaveData(outpath)


def run_orca(sim: Sim) -> None:
    # Runs an Orcaflex simulation, writes results to file
    base_path = PATH / sim.base_filename

    model = ofx.Model(base_path)

    all_names = [obj.Name for obj in model.objects]

    ovessel = [obj for obj in model.objects if obj.typeName == 'Vessel'][0]
    vtype = model[ovessel.VesselType]

    env = model.environment

    # Wave
    env.SelectedWaveTrain = 'Wave1'
    env.WaveHs = sim.hs
    env.WaveGamma = sim.gamma
    env.WaveTp = sim.tp
    env.WaveDirection = sim.dirn
    env.WaveSeed = sim.seed
    env.WaveOriginX = ovessel.InitialX - vtype.Length / 2
    env.WaveTimeOrigin = sim.t_origin

    # Current
    if sim.cspd == 1.0:
        env.RefCurrentSpeed = sim.cspd
        env.RefCurrentDirection = sim.cdirn
        df = pd.read_csv(PATH / 'current_profiles.csv', header=[0, 1])
        cprofile = df[f'profile {sim.cprofile}'].dropna()
        env.NumberOfCurrentLevels = len(cprofile.index)
        env.CurrentDepth = cprofile['depth']
        env.CurrentFactor = cprofile['speed']

    model.general.StageDuration[1] = sim.duration

    print('Running files LC_{0:05d}'.format(sim.lc))

    if sim.save_dat:
        outpath = PATH / 'rerun' / f'LC_{sim.lc:05d}.dat'
        model.SaveData(outpath)

    if not sim.run_sim:
        return

    success = False
    for t_step in [0.1, 0.05, 0.02]:
        model.general.ImplicitConstantTimeStep = t_step
        model.RunSimulation()
        if model.state.value == 4:
            success = True
            break
    if not success:
        return

    # RESULTS ----------------------------------------------------
    results = [sim.lc]

    # PIPELINE RESULTS --------------------------------------------------
    line = model['Line1']
    ltype = model[line.LineType[0]]

    line_length = line.CumulativeLength[-1]

    roller_names = [x[3:] for x in all_names if x[:5] in ['b6 BR', 'b6 SR']]
    last_roller = model['b6 ' + roller_names[-1]]
    last_roller_arc = float(last_roller.tags['arc']) / 1000

    # Layback
    s_lay = line.StaticResult('Layback', ofx.oeEndA) - line.EndAX
    d_lay = line.TimeHistory('Layback', 1, ofx.oeEndA) - line.EndAX
    results += [s_lay, d_lay.max(), d_lay.min()]

    # Suspended length from transom
    var, objx = 'Arc length', ofx.oeTouchdown
    s_len = line.StaticResult(var, objx) - line.EndAX
    d_len = line.TimeHistory(var, 1, objx) - line.EndAX
    results += [s_len, d_len.max(), d_len.min()]

    # Top tension
    var, objx = 'Effective Tension', ofx.oeEndA
    static = line.StaticResult(var, objx)
    dyn = line.TimeHistory(var, 1, objx)
    results += [static, dyn.max(), dyn.min()]

    # TDP tension
    var, objx = 'Effective Tension', ofx.oeTouchdown
    static = line.StaticResult(var, objx)
    dyn = line.TimeHistory(var, 1, objx)
    results += [static, dyn.max(), dyn.min()]

    # Stinger tip clearance
    var = 'Support contact clearance'
    clr1 = last_roller.StaticResult(var, ofx.oeSupport(1))
    clr2 = last_roller.StaticResult(var, ofx.oeSupport(2))
    static = (clr1 + clr2) / 2
    clr1 = last_roller.TimeHistory(var, 1, ofx.oeSupport(1))
    clr2 = last_roller.TimeHistory(var, 1, ofx.oeSupport(2))
    dyn = (clr1 + clr2) / 2
    results += [static, dyn.max(), dyn.min()]

    # Roller loads
    for roller_type in 'BR', 'SR':
        roller_names = [x[3:] for x in all_names
                        if x[:5] == f'b6 {roller_type}']
        static, dyn = get_roller_loads(roller_names, model)
        results += [static, dyn]

    # CODE CHECKS -----------------------------
    # Functional / Environmental
    # a -> 1.2 / 0.7
    # b -> 1.1 / 1.3
    codechecks = model['Code checks']
    ltype.DNVSTF101AlphaPm = 1.0

    arc_ob = ofx.arSpecifiedArclengths(0, last_roller_arc - 10)
    arc_st = ofx.arSpecifiedArclengths(last_roller_arc - 10,
                                       last_roller_arc + 10)
    arc_sb = ofx.arSpecifiedArclengths(last_roller_arc + 10,
                                       line_length)

    for gamma_f, gamma_e in [[1.2, 0.7], [1.1, 1.3]]:
        codechecks.DNVSTF101GammaF = gamma_f
        codechecks.DNVSTF101GammaE = gamma_e
        var = 'DNV OS F101 load controlled'
        for arc, gamma_c in [[arc_ob, 0.8], [arc_st, 1.0], [arc_sb, 1.0]]:
            codechecks.DNVSTF101GammaC = gamma_c
            static = line.RangeGraph(var, ofx.pnStaticState, None, arc).Mean
            dyn = line.RangeGraph(var, 1, None, arc).Max
            results += [static.max(), dyn.max()]

    outpath = PATH / 'sims' / f'LC_{sim.lc:05d}.txt'
    with open(outpath, 'w') as f:
        f.write('\t'.join('{0:.5e}'.format(x) for x in results) + '\n')

    # model.SaveSimulation('{0}\\sims\\LC_{1:05d}.sim'.format(folder, lc))


def combine_results() -> None:

    txtpaths = (PATH / 'sims').glob('*.txt')
    res = []
    for pth in txtpaths:
        with open(pth, 'r') as f:
            for line in f:
                res.append([float(x) for x in line.split()])

    header = get_header()
    df = pd.DataFrame(res, columns=header)

    sims = pd.read_excel(PATH / 'sims.xlsx')
    df2 = sims.merge(df)

    df2.to_excel(PATH / 'results.xlsx', index=False)


def run_sims(ncpu: int, rerun: list[int] | None = None):

    fpath = PATH / 'sims'
    if not fpath.exists():
        fpath.mkdir()

    df = pd.read_excel(PATH / 'sims.xlsx')

    sims: list[Sim] = []
    for dic in df.to_dict('records'):
        sims.append(Sim(**dic))

    if rerun:
        fpath = PATH / 'rerun'
        if not fpath.exists():
            fpath.mkdir()
        for lc in rerun:
            sims[lc - 1].save_dat = True
            sims[lc - 1].run_sim = False
        sims = [sims[lc - 1] for lc in rerun]

    nsim = len(sims)

    print('Starting {0} sims with {1} processors'.format(nsim, ncpu))

    with Pool(ncpu) as p:
        p.map(run_orca, sims, chunksize=1)

    combine_results()


def result_summary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    # Columns
    cols_static = ['dirn_name']
    cols_dyn = ['dirn_name']
    for col in cols:
        if col + ' max' in df.columns:
            cols_static.append(col + ' static')
            cols_dyn.append(col + ' max')
        if col + ' min' in df.columns:
            cols_static.append(col + ' static')
            cols_dyn.append(col + ' min')

    # Static
    # cols_static = ['dirn_name'] + [x + ' static' for x in cols]
    res0 = df[cols_static]
    res0.columns = cols_dyn
    res0.loc[:, 'dirn_name'] = 'static'

    res1 = df[cols_dyn]
    # res3.columns = ['dirn_name'] + cols

    res2 = pd.concat((res0, res1)).drop_duplicates()

    return res2


def postprocess() -> None:

    df = pd.read_excel(PATH / 'results.xlsx')

    # Take average of 5 seeds
    aggs = {'dirn_name': 'max'}
    for col in df.columns:
        if col.split()[-1] in ['static', 'max', 'min']:
            aggs[col] = 'mean'
    res0 = df.groupby(['hs', 'tp', 'dirn']).agg(aggs)
    res0.reset_index(inplace=True)
    res0.sort_values(['tp', 'dirn'], inplace=True)
    # res0.to_excel(PATH / 'summary_1.xlsx', index=False)
    row_order = res0['dirn_name'].unique().tolist()

    # Group by dirn_name
    lookup = {'static': 'max', 'max': 'max', 'min': 'min'}
    aggs = {}
    for col in df.columns:
        if col.split()[-1] in lookup.keys():
            aggs[col] = lookup[col.split()[-1]]
    res1 = res0.groupby('dirn_name').agg(aggs)
    res1 = res1.loc[row_order]
    res1.reset_index(inplace=True)
    # res1.to_excel(PATH / 'summary_2.xlsx', index=False)

    # Layback and tensions
    cols = ['layback', 'susp_length', 'top_tension', 'tdp_tension']
    res2 = result_summary(res1, cols)
    # res2.to_excel(PATH / 'layback.xlsx', index=False)

    # Tip clearance, roller loads
    cols = ['tip_clearance', 'barge_roller_load', 'stinger_roller_load']
    res3 = result_summary(res1, cols)
    # res3.to_excel(PATH / 'roller_loads.xlsx', index=False)

    # F101 results summary
    cols = []
    for lc in ['a', 'b']:
        for loc in ['overbend', 'tip', 'sag']:
            cols.append(f'f101{lc}_{loc}')
    res4 = result_summary(res1, cols)
    # res4.to_excel(PATH / 'f101.xlsx', index=False)

    with pd.ExcelWriter(PATH / 'summary.xlsx') as writer:
        res0.to_excel(writer, sheet_name='Avg 5 seeds', index=False)
        res1.to_excel(writer, sheet_name='Group by dirn', index=False)
        res2.to_excel(writer, sheet_name='Layback, tension', index=False)
        res3.to_excel(writer, sheet_name='Roller loads', index=False)
        res4.to_excel(writer, sheet_name='F101', index=False)

def main():
    pass

if __name__ == "__main__":
    main()
