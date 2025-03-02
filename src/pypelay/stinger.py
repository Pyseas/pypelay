
from .geom import *
import datetime
from pathlib import Path
import OrcFxAPI as ofx
import math
import pandas as pd
import numpy as np
import warnings
from dataclasses import dataclass, field
from openpyxl import load_workbook
from openpyxl.styles import Alignment
from importlib.resources import files

PATH = Path('.')

__all__ = ['static_summary', 'set_radius', 'select_radius', 'stinger_setup',
           'adjust_top_tension', 'Vessel']


@dataclass
class Roller:
    name: str  # SR1, SR2 etc
    y: float = 0.0  # Roller height (in mm)
    r3: float = 0.0  # Angle in deg relative to stinger section bottom chord
    post_angle: float = 0.0  # Post angle in deg relative to vessel
    arc: float = 0.0  # Arc length along pipe path (in mm)
    y_offset: float = 0.0  # Offset from V to pipe CL (in mm)

@dataclass
class Vessel:
    name: str = 'S1200'
    draft: float = 7400

@dataclass
class StingerSetupArgs:
    inpath: Path
    outpath: Path
    straight: float
    transition: float
    ang1: float
    ang2: float
    water_depth: float
    tip_clearance: float
    delete_dat: bool = False

@dataclass
class StingerSetupOptions:
    tip_def: str
    tip_len: float
    seglen: float
    ob_seglen: float
    tdp_seglen: float
    tdp_len: float
    tensioner_mode: str
    deadband: float

@dataclass
class StingerSetupResults:
    outpath: Path
    top_tension: float = -1.0
    uc_ob: float = -1.0
    uc_sag: float = -1.0
    tip_depth: float = 0.0
    tip_angle: float = 0.0
    draft: float = 0.0

    def __str__(self):
        outstr = f'{self.outpath.name} : '
        outstr += f'top_tension {self.top_tension:.1f}kN, '
        outstr += f'uc_sag {self.uc_sag:.3f}, '
        outstr += f'tip_depth {self.tip_depth:.1f}m'

        return outstr

@dataclass
class LineType:
    Name: str
    OD: float
    ID: float
    CoatingThickness: float
    CoatingMaterialDensity: float
    LiningThickness: float
    LiningMaterialDensity: float
    WallThickness: float = field(init=False)

    def __post_init__(self):
        self.WallThickness = (self.OD - self.ID) / 2

    def weights(self, contents_density: float) -> tuple[float, float]:
        pipe_wt = np.pi / 4 * (self.OD**2 - self.ID**2) * 7.85
        # Coating
        coating_od = self.OD + 2 * self.CoatingThickness
        coating_id = self.OD
        coating_wt = (np.pi / 4 * (coating_od**2 - coating_id**2) *
                      self.CoatingMaterialDensity)
        # Lining
        lining_od = self.ID
        lining_id = self.ID - 2 * self.LiningThickness
        lining_wt = (np.pi / 4 * (lining_od**2 - lining_id**2) *
                      self.LiningMaterialDensity)

        wt_in_air = pipe_wt + coating_wt + lining_wt
        contents_wt = np.pi / 4 * lining_id**2 * contents_density
        disp = np.pi / 4 * coating_od**2 * 1.025
        wt_submerged = wt_in_air - disp - contents_wt

        return wt_in_air, wt_submerged


def static_summary(outpath, datpaths: list[Path]):

    xlpath = files('pypelay') / (f'static_summary.xlsx')
    wb = load_workbook(xlpath)
    ws = wb['Sheet1']
    # style_str = NamedStyle(name='style_str')
    # style_str.alignment = Alignment(vertical='center', horizontal='center')

    icol = 3
    for dpath in datpaths:

        model = ofx.Model(dpath)
        model.CalculateStatics()

        all_names = [obj.Name for obj in model.objects]

        stinger_ref = model['b6 stinger_ref']

        radius = float(stinger_ref.tags['radius']) / 1000
        num_section = int(stinger_ref.tags['num_section'])

        vname = stinger_ref.Connection
        ovessel = model[vname]
        bollard_pull = ovessel.GlobalAppliedForceX[0]

        # Pipe data
        oltype = [obj for obj in model.objects
                 if obj.typeName == 'Line type'][0]
        ltype = LineType(
            oltype.Name, oltype.OD, oltype.ID,
            oltype.CoatingThickness, oltype.CoatingMaterialDensity,
            oltype.LiningThickness, oltype.LiningMaterialDensity)
        wt_in_air, wt_submerged = ltype.weights(0.0)

        roller_names = [x[3:] for x in all_names if x[:5] in ['b6 BR', 'b6 SR']]
        last_roller = model['b6 ' + roller_names[-1]]
        last_roller_arc = float(last_roller.tags['arc']) / 1000

        line = model['Line1']
        top_tension = float(line.StaticResult('Effective tension', ofx.oeEndA))

        # Layback, scope, gain
        firing_line = model['b6 firing_line']
        beadstall = float(firing_line.tags['bead stall'])
        bstall_x = firing_line.StaticResult('X', ofx.oeBuoy(beadstall, 0, 0))
        tdp_x = line.StaticResult('X', ofx.oeTouchdown)
        enda_x = line.StaticResult('X', ofx.oeEndA)
        tdp_arc = line.StaticResult('Arc length', ofx.oeTouchdown)
        layback = float(bstall_x - tdp_x)
        scope_length = float(tdp_arc + (bstall_x - enda_x))
        pipe_gain = scope_length - layback

        # Tip clearance
        clear = 0
        for isup in [1, 2]:
            objx = ofx.oeSupport(isup, 'Line1')
            clear += last_roller.StaticResult(
                'Support contact clearance', objx)
        tip_clearance = float(clear) / 2

        # Stinger draft and seabed clearance
        last_section = model[f'b6 stinger_{num_section}']
        vertx = min(last_section.VertexX)
        objx = ofx.oeBuoy(vertx, 0, 0)
        draft = float(last_section.StaticResult('Z', objx)) * -1
        depth = model.environment.WaterDepth
        seabed_clearance = depth - draft

        # Code checks
        line_length = line.CumulativeLength[-1]
        arc_ob = ofx.arSpecifiedArclengths(0, last_roller_arc)
        arc_sag = ofx.arSpecifiedArclengths(last_roller_arc, line_length)

        code_checks = model['Code checks']
        code_checks.DNVSTF101GammaF = 1.2
        code_checks.DNVSTF101GammaE = 0.7

        # Overbend
        code_checks.DNVSTF101GammaC = 0.8
        vars = ['DNV ST F101 load controlled',
                'DNV ST F101 disp. controlled',
                'Worst ZZ strain']
        res_ob = []
        for var in vars:
            res_ob.append(line.RangeGraph(var, None, None, arc_ob).Mean.max())

        # Sag bend
        code_checks.DNVSTF101GammaC = 1.0
        vars = ['DNV ST F101 load controlled', 'Max von Mises stress']
        res_sag = []
        for var in vars:
            res_sag.append(line.RangeGraph(var, None, None, arc_sag).Mean.max())

        # Rollers
        support_loads = []
        roller_depths = []
        for rname in roller_names:
            oroller = model[f'b6 {rname}']
            nsup = oroller.NumberOfSupports
            react = 0
            for isup in range(nsup):
                objx = ofx.oeSupport(isup + 1)
                react += oroller.StaticResult('Support reaction force', objx)
            support_loads.append(react)
            roller_depths.append(oroller.StaticResult('Z'))

        ws.cell(2, icol).value = dpath.stem
        ws.cell(3, icol).value = ' '.join(vname.split()[1:])
        ws.cell(4, icol).value = model.environment.WaterDepth

        ws.cell(7, icol).value = radius
        ws.cell(8, icol).value = num_section

        ws.cell(11, icol).value = ltype.OD * 1000
        ws.cell(12, icol).value = ltype.WallThickness * 1000
        ws.cell(13, icol).value = ltype.CoatingThickness * 1000
        ws.cell(14, icol).value = ltype.CoatingMaterialDensity * 1000
        ws.cell(15, icol).value = ltype.LiningThickness * 1000
        ws.cell(16, icol).value = ltype.LiningMaterialDensity * 1000
        ws.cell(17, icol).value = wt_in_air * 1000
        ws.cell(18, icol).value = wt_submerged * 1000

        # Static results
        ws.cell(21, icol).value = bollard_pull
        ws.cell(22, icol).value = top_tension
        ws.cell(23, icol).value = layback
        ws.cell(24, icol).value = scope_length
        ws.cell(25, icol).value = pipe_gain
        ws.cell(26, icol).value = tip_clearance
        ws.cell(27, icol).value = draft
        ws.cell(28, icol).value = seabed_clearance

        # Pipe loads
        ws.cell(31, icol).value = res_ob[0]
        ws.cell(32, icol).value = res_ob[1]
        ws.cell(33, icol).value = res_ob[2]
        ws.cell(34, icol).value = res_sag[0]
        ws.cell(35, icol).value = res_sag[1] / 1000  # vm stress

        irow = 38
        for load in support_loads:
            ws.cell(irow, icol).value = load
            irow += 1

        irow = 54
        for depth in roller_depths:
            ws.cell(irow, icol).value = depth
            irow += 1

        # Set cell alignment
        for irow in range(1, 80):
            ws.cell(irow, icol).alignment = Alignment(
                horizontal='center', vertical='center')

        # ws.cell(17, icol).number_format = '0.0'
        # ws.cell(18, icol).number_format = '0.0'

        icol += 1

    wb.save(outpath)


def set_radius(vessel: Vessel, num_section: int, radius: float,
               water_depth: float, tip_clearance: float,
               outpath: Path) -> None:

    radius *= 1000
    xlpath = files('pypelay') / f'{vessel.name}_configs.xlsx'
    df = pd.read_excel(xlpath)

    df = df[(df['num_section'] == num_section) & 
            (df['prefer'] == 1)]

    df.sort_values(['radius'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Find closest radius
    ind = (df['radius'] - radius).abs().argmin()
    row = df.iloc[ind].to_dict()

    model = get_base_case(vessel, row['radius'], num_section)

    stargs = StingerSetupArgs(
        PATH / 'base case.dat', outpath,
        row['straight'], row['transition'], row['ang1'], row['ang2'],
        water_depth, tip_clearance, delete_dat=False
    )

    res = stinger_setup(stargs)

    (PATH / 'base case.dat').unlink()


def select_radius(vessel: Vessel, num_section: int,
                  water_depth: float, tip_clearance: float,
                  lcc_target: float):

    # STAGE 1: Use simple catenary model to get first estimate of
    # stinger radius
    vmodel = ofx.Model()
    ovessel = vmodel.CreateObject(ofx.ObjectType.Vessel, name='Vessel1')
    vmodel.environment.Depth = water_depth

    # Fetch pipe linetype from pipe dat file
    datpath = PATH / 'pipe.dat'
    pmodel = ofx.Model(datpath)  # Pipe model
    linetypes = [obj.Name for obj in pmodel.objects if 
                    obj.typeName == 'Line type']
    linetype = pmodel[linetypes[0]]
    clone = linetype.CreateClone(name=linetype.Name, model=vmodel)
    line = vmodel.CreateObject(ofx.ObjectType.Line, name='Line1')

    code_checks = vmodel['Code checks']
    code_checks.DNVSTF101GammaF = 1.2
    code_checks.DNVSTF101GammaE = 0.7
    code_checks.DNVSTF101GammaC = 1.0

    line.EndAConnection = 'Vessel1'
    line.EndAX = 0.0
    line.EndAY = 0.0
    line.EndAZ = 0.0
    line.LayAzimuth = 0.0
    line.TargetSegmentLength[0] = 5.0

    # First estimate of top angle used to set line length
    # Should use a formula instead (top_angle = f(water_depth))
    if water_depth < 100:
        top_angle = math.radians(30.0)
    elif 100 <= water_depth <= 500:
        top_angle = math.radians(50.0)
    else:
        top_angle = math.radians(70.0)

    dh = water_depth
    susp_len, dx = catenary_length(dh, top_angle)
    length_on_seabed = 200
    line.EndBConnection = 'Anchored'
    line.EndBX = -(dx + length_on_seabed)
    line.EndBHeightAboveSeabed = 0.0
    line.Length[0] = susp_len + length_on_seabed

    vmodel.CalculateStatics()

    lcc0 = line.RangeGraph('DNV OS F101 load controlled').Mean.max()

    bp0 = -ovessel.StaticResult('Connections GX force')
    if bp0 < 10:
        bp0 = 10
    bp1 = bp0 * 1.05

    ovessel.IncludedInStatics = '3 DOF'
    ovessel.IncludeAppliedLoads = 'Yes'
    ovessel.NumberOfGlobalAppliedLoads = 1
    ovessel.GlobalAppliedLoadOriginX[0] = 50.0
    ovessel.GlobalAppliedForceX[0] = bp1

    # Iterate bollard pull to get target sag bend UC
    lcc1 = 0.0
    for _ in range(10):
        ovessel.GlobalAppliedForceX[0] = bp1
        vmodel.CalculateStatics()
        lcc1 = line.RangeGraph('DNV OS F101 load controlled').Mean.max()
        if abs(lcc1 - lcc_target) < 0.005:
            break
        bp2 = (lcc_target - lcc0) / (lcc1 - lcc0) * (bp1 - bp0) + bp0
        bp2 = min([float(x) for x in [bp1 * 2, bp2]])
        bp2 = max([float(x) for x in [bp1 * 0.5, bp2]])
        bp0, bp1 = bp1, bp2
        lcc0 = lcc1

    arc_ang = line.RangeGraphXaxis('Declination')
    line_angles = line.RangeGraph('Declination').Mean + 90
    arc_dep = line.RangeGraphXaxis('Z')
    line_depths = line.RangeGraph('Z').Mean
    line_angles = np.interp(arc_dep, arc_ang, line_angles)

    xlpath = files('pypelay') / (f'{vessel.name}_configs.xlsx')
    # xlpath = PATH / f'{vessel.name}_configs.xlsx'
    df = pd.read_excel(xlpath)

    df = df[(df['num_section'] == num_section) & 
            (df['prefer'] == 1)]

    df.sort_values(['radius'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Stinger radius is determined from line angle at stinger tip
    # Need to iterate since stinger tip angle varies with tip depth
    radii = df[f'radius'].to_numpy()
    tip_angles = df[f'tip_angle'].to_numpy()
    tip_depths = df[f'tip_depth'].to_numpy()

    top_angle = line_angles[0]
    for _ in range(5):
        rad_mm = float(np.interp(-top_angle, -tip_angles, radii))
        tip_depth = float(np.interp(rad_mm, radii, tip_depths))
        top_angle = float(np.interp(-tip_depth, -line_depths, line_angles))

    ind = (df['radius'] - rad_mm).abs().argmin()

    lc = 1
    toggle = 0
    radii = [0, 0]
    for i in range(5):
        row = df.iloc[ind].to_dict()
        radius = row['radius']

        model = get_base_case(vessel, row['radius'], num_section)

        outpath = PATH / f'tmp{toggle}.dat'
        radii[toggle] = radius
        toggle = 1 - toggle

        stargs = StingerSetupArgs(
            PATH / 'base case.dat', outpath,
            row['straight'], row['transition'], row['ang1'], row['ang2'],
            water_depth, tip_clearance, delete_dat=False
        )

        res = stinger_setup(stargs)

        (PATH / 'base case.dat').unlink()

        lcc1 = res.uc_sag 
        if lcc1 > lcc_target:
            ind += 1
        else:
            ind += -1

        if i > 0:
            if min(lcc0, lcc1) < lcc_target < max(lcc0, lcc1):
                to_keep = toggle if lcc0 < lcc1 else 1 - toggle
                to_del = 1 - to_keep
                (PATH / f'tmp{to_del}.dat').unlink()
                keep_path = PATH / f'R{radii[to_keep] / 1000:.0f}.dat'
                if keep_path.exists():
                    inp = input(f'File {keep_path.name} already exists. Overwrite (y/n)? ')
                    if inp.lower() == 'y':
                        keep_path.unlink()
                    else:
                        (PATH / f'tmp{to_keep}.dat').unlink()
                        break
                (PATH / f'tmp{to_keep}.dat').rename(keep_path)
                break

        lcc0 = lcc1
        lc += 1


def stinger_tip_data(model: ofx.Model) -> tuple[float, float, float]:

    stinger_ref = model['b6 stinger_ref']
    num_section = int(stinger_ref.tags['num_section'])

    last_section = model[f'b6 stinger_{num_section}']
    all_names = [obj.Name for obj in model.objects]
    roller_names = [x[3:] for x in all_names if x[:5] in ['b6 BR', 'b6 SR']]
    last_roller = model['b6 ' + roller_names[-1]]

    model.CalculateStatics()

    vertx = min(last_section.VertexX)
    objx = ofx.oeBuoy(vertx, 0, 0)
    draft = float(last_section.StaticResult('Z', objx))
    # tip angle is equal to stinger section angle plus roller r3
    conn = last_section.Connection
    last_section.Connection = 'Fixed'
    section_angle = float(last_section.InitialRotation3 + 180)
    last_section.Connection = conn

    conn = last_roller.Connection
    last_roller.Connection = 'Fixed'
    tip_depth = float(last_roller.InitialZ)
    last_roller.Connection = conn

    roller_r3 = float(last_roller.tags['r3'])
    tip_angle = section_angle + roller_r3

    model.CalculateStatics()

    return tip_depth, tip_angle, draft


def get_options() -> StingerSetupOptions:

    # Ignore Data validation warning
    with warnings.catch_warnings(action='ignore', category=UserWarning):
        df = pd.read_excel(PATH / 'options.xlsx', sheet_name='Sheet1')

    df.set_index('ID', inplace=True)
    opts = StingerSetupOptions(
        tip_def = df.loc[1, 'Value'],
        tip_len = df.loc[2, 'Value'],
        seglen = df.loc[3, 'Value'],
        ob_seglen = df.loc[4, 'Value'],
        tdp_seglen = df.loc[5, 'Value'],
        tdp_len = df.loc[6, 'Value'],
        tensioner_mode = df.loc[7, 'Value'],
        deadband = df.loc[8, 'Value'],
    )

    return opts


def stinger_setup(sim: StingerSetupArgs) -> StingerSetupResults:
    '''
     - Calculate pipe path and roller heights
     - Set roller heights
     - Create pipe
     - Adjust bollard pull to get target tip clearance
     - Update segmentation around TDP
     - Add deadband winch if needed
    '''

    model = ofx.Model(sim.inpath)
    # print(sim.outpath)

    # Read options spreadsheet
    opts = get_options()

    model.environment.WaterDepth = sim.water_depth

    # Set stinger section angles
    stinger_ref = model['b6 stinger_ref']
    num_section = int(stinger_ref.tags['num_section'])
    radius = int(stinger_ref.tags['radius'])
    model['b6 stinger_1'].InitialRotation3 = sim.ang1
    if num_section > 1:
        model['b6 stinger_2'].InitialRotation3 = sim.ang2

    # Set roller heights and angles
    path_coords = calc_path_coords(model, sim.straight, sim.transition)
    rollers = get_roller_heights(model, path_coords, sim.ang1, sim.ang2)

    if not rollers:
        print('Stinger config not valid, roller heights outside limits')
        return StingerSetupResults(sim.outpath)

    for roller in rollers:
        oroller = model[f'cn {roller.name}']
        oroller.InitialY = roller.y / 1000
        oroller.DOFInitialValue[5] = roller.r3

    # Create line
    roller = rollers[-1]
    # Connect line to roller at y_offset then change connection to stinger_ref
    line = model.CreateObject(ofx.ObjectType.Line, 'Line1')
    line.LayAzimuth = 0.0
    line.NumberOfSections = 2
    # Line overbend (first section) extends 10m past last roller
    line.Length[0] = roller.arc / 1000 + 10.0
    line.TargetSegmentLength[0] = opts.ob_seglen
    line.TargetSegmentLength[1] = opts.seglen
    line.FullStaticsMinDamping = 5.0
    line.FullStaticsMaxDamping = 20.0

    buoy_name = f'b6 {roller.name}'
    line.EndAConnection = buoy_name
    line.EndAX = 0.0
    line.EndAY = roller.y_offset / 1000
    line.EndAZ = 0.0
    line.EndAConnection = 'Fixed'
    end_a = np.array([line.EndAX, line.EndAZ])
    line.EndAConnection = 'b6 stinger_ref'

    top_angle = math.radians(roller.post_angle + roller.r3 - 90)
    dh = end_a[1] + sim.water_depth
    susp_len, dx = catenary_length(dh, top_angle)
    length_on_seabed = 200
    line.EndBConnection = 'Anchored'
    line.EndBX = end_a[0] - dx - length_on_seabed
    line.EndBHeightAboveSeabed = 0.0

    line.Length[1] = susp_len + length_on_seabed
    line.EndAConnection = 'b6 stinger_ref'
    line.EndAX = path_coords[-1, 0] / 1000
    line.EndAY = path_coords[-1, 1] / 1000
    # Set top end orientation and stiffness
    line.IncludeTorsion = 'Yes'
    line.EndAxBendingStiffness = ofx.OrcinaInfinity()
    line.EndATwistingStiffness = ofx.OrcinaInfinity()
    line.EndAAzimuth = 180
    line.EndADeclination = 90
    line.EndAGamma = 180

    # Set rollers supported line
    model['b6 firing_line'].NumberOfSupportedLines = 1
    model['b6 firing_line'].SupportedLine[0] = 'Line1'
    for roller in rollers:
        buoy = model[f'b6 {roller.name}']
        buoy.NumberOfSupportedLines = 1
        buoy.SupportedLine[0] = 'Line1'
        buoy.tags['post_angle'] = f'{roller.post_angle:.1f}'
        buoy.tags['r3'] = f'{roller.r3:.3f}'
        buoy.tags['arc'] = f'{roller.arc:.1f}'

    # Solve statics, use calculated line shapes
    solved = False
    for _ in range(5):
        try:
            model.CalculateStatics()
            model.UseCalculatedPositions(SetLinesToUserSpecifiedStartingShape=True)
            solved = True
            break
        except:
            newdamp = line.FullStaticsMinDamping + 2.0
            line.FullStaticsMinDamping = newdamp
            model.SaveData(sim.outpath)
            print(f'Loadcase {sim.outpath.name} increasing damping to {newdamp}')

    if not solved:
        print(f'Loadcase {sim.outpath.name} failed to solve')
        return StingerSetupResults(sim.outpath)

    # Write stinger config details to b6 stinger_ref tags,
    # add comment to General->Comments
    buoy = model['b6 stinger_ref']
    buoy.tags['date'] = datetime.datetime.now().strftime('%Y-%m-%d')
    # buoy.tags['pipe_od'] = f'{pipe_od:.1f}'
    buoy.tags['straight'] = f'{sim.straight:.1f}'
    buoy.tags['transition'] = f'{sim.transition:.1f}'

    outstr = f'Stinger configuration generated with Python pipelay module.\n'
    outstr += 'For details refer to "b6 stinger_ref" tags.'
    model.general.Comments = outstr

    # datpath = PATH / 'sims' / f'LC_{sim.lc:05d}.dat'
    # model.SaveData(sim.outpath)

    # Adjust bollard pull to get target tip clearance
    # (updates model, unsolved state)
    solved = set_bollard_pull(model, 'tip_clearance', sim.tip_clearance, tol=0.01)
    if not solved:
        print('Failed to converge (tip clearance)')
        return StingerSetupResults(sim.outpath)

    update_segmentation(model, opts)

    if opts.tensioner_mode != 'Brake':
        add_deadband_winch(model, opts)

    model.SaveData(sim.outpath)

    # Create dat file suitable for running dynamics (vessel )
    prep_for_dyn(sim.outpath)

    res = get_setup_results(model, sim.outpath)

    # If running all 18000 configs need to delete dat files
    if sim.delete_dat:
        sim.outpath.unlink()

    print(f'Radius {radius / 1000:.0f}m : {res}')

    return res


def prep_for_dyn(datpath: Path) -> None:
    ''' 
    Prepare model for dynamic analysis:
     - Set roller orientations, delete constraints
     - Switch off vessel 3 DOF (so that current doesn't effect position)
    '''
    model = ofx.Model(datpath)

    fix_rollers(model)

    ovessel = [obj for obj in model.objects if obj.typeName == 'Vessel'][0]
    ovessel.IncludedInStatics = 'None'

    outpath = datpath.parent / f'{datpath.stem}_dyn.dat'
    model.SaveData(outpath)


def fix_rollers(model: ofx.Model) -> None:

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

 
def add_deadband_winch(model: ofx.Model, opts: StingerSetupOptions) -> None:

    if model.state == 0:
        # Reset state = 0, InStaticState = 2
        model.CalculateStatics()

    line = model['Line1']
    top_tension = line.StaticResult('Effective tension', ofx.oeEndA)
    match opts.tensioner_mode:
        case 'Deadband: Default':
            if top_tension <= 100 * 9.81:
                deadband = 10 * 9.81
            else:
                deadband = top_tension * 0.1
        case 'Deadband: % top tension':
            deadband = top_tension * opts.deadband
        case 'Deadband: tension in t':
            deadband = opts.deadband * 9.81

    cn = model.CreateObject(ofx.ObjectType.Constraint, name='cn tensioner')
    cn.Connection = 'b6 stinger_ref'
    cn.InFrameInitialX = line.EndAX
    cn.InFrameInitialY = line.EndAY
    cn.InFrameInitialAzimuth = 180
    cn.InFrameInitialDeclination = 90
    cn.InFrameInitialGamma = 180
    cn.DOFFree[2] = 'Yes'
    cn.DOFFree[3] = 'Yes'

    winch = model.CreateObject(ofx.ObjectType.Winch, name='w tensioner')
    winch.WinchType = 'Detailed'
    winch.Connection[0] = 'b6 stinger_ref'
    # x = top_tension * L / EA
    winch.ConnectionX[0] = line.EndAX + 10.0 + top_tension * 1E-4
    winch.ConnectionY[0] = line.EndAY
    winch.ConnectionZ[0] = 0.0
    winch.Connection[1] = 'cn tensioner'
    winch.ConnectionX[1] = 0.0
    winch.ConnectionY[1] = 0.0
    winch.ConnectionZ[1] = 0.0
    winch.WinchControlType = 'Whole simulation'
    winch.StaticMode = 'Specified length'
    winch.StaticValue = 10.0
    winch.WholeSimulationControlMode = 'Specified tension'
    winch.WholeSimulationTension = top_tension
    winch.Stiffness = 100E3
    winch.Damping = 0.05
    winch.DriveDeadband = deadband
    winch.DriveDampingHaulIn = 0.1
    winch.DriveDampingPayOut = 0.1

    line.EndAConnection = 'cn tensioner'
    line.EndAAzimuth = 0.0
    line.EndADeclination = 0.0
    line.EndAGamma = 0.0


def update_segmentation(model: ofx.Model, opts: StingerSetupOptions) -> None:

    if model.state == 0:
        # Reset state = 0, InStaticState = 2
        model.CalculateStatics()

    line = model['Line1']

    tdp_arc = line.StaticResult('Arc length', ofx.oeTouchdown)

    tdp_len = opts.tdp_len

    # Section 1 is overbend, from start of line to 10m past last roller
    # (does not change)
    # Section 2 is up to TDP section
    # Section 3 is TDP region
    # Section 4 is from TDP region to end of line

    len0 = line.CumulativeLength[0]
    line_length = line.CumulativeLength[-1]

    extends_to_tip = False
    if (tdp_arc - tdp_len / 2) < (len0 + 2):
        extends_to_tip = True

    extends_to_end = False
    if tdp_arc + tdp_len / 2 > line_length - 5:
        extends_to_end = True

    if all([extends_to_tip, extends_to_end]):
        line.NumberOfSections = 2
        line.Length[1] = line_length - len0
        line.TargetSegmentLength[1] = opts.tdp_seglen

    if all([extends_to_tip, not extends_to_end]):
        line.NumberOfSections = 3
        line.Length[1] = tdp_arc + tdp_len / 2 - len0
        line.TargetSegmentLength[1] = opts.tdp_seglen
        line.Length[2] = line_length - (tdp_arc + tdp_len / 2)
        line.TargetSegmentLength[2] = opts.seglen

    if all([not extends_to_tip, extends_to_end]):
        line.NumberOfSections = 3
        line.Length[1] = tdp_arc - tdp_len / 2 - len0
        line.TargetSegmentLength[1] = opts.seglen
        line.Length[2] = line_length - (tdp_arc - tdp_len / 2)
        line.TargetSegmentLength[2] = opts.tdp_seglen

    if all([not extends_to_tip, not extends_to_end]):
        line.NumberOfSections = 4
        line.Length[1] = tdp_arc - tdp_len / 2 - len0
        line.TargetSegmentLength[1] = opts.seglen
        line.Length[2] = tdp_len
        line.TargetSegmentLength[2] = opts.tdp_seglen
        line.Length[3] = line_length - (tdp_arc + tdp_len / 2)
        line.TargetSegmentLength[3] = opts.seglen

    line.StaticsStep1 = 'Catenary'

    solved = False
    for _ in range(5):
        try:
            model.CalculateStatics()
            model.UseCalculatedPositions(SetLinesToUserSpecifiedStartingShape=True)
            solved = True
            break
        except:
            newdamp = line.FullStaticsMinDamping + 2.0
            line.FullStaticsMinDamping = newdamp
            print(f'Increasing line damping to {newdamp}')

    if not solved:
        print('Couldnt solve after updating segmentation')


def adjust_top_tension(inpath: Path, outpath: Path, tension) -> None:

    model = ofx.Model(inpath)

    model.SaveData(outpath)

    solved = set_bollard_pull(model, 'top_tension', tension, tol=1.0)

    if not solved:
        print('Failed to converge (top tension)')
        return

    opts = get_options()
    update_segmentation(model, opts)

    model.SaveData(outpath)

    res = get_setup_results(model, outpath)

    print(res)


def set_bollard_pull(model: ofx.Model, 
                     target_var: str, target_val: float,
                     tol: float) -> bool:
    # Iterates vessel bollard pull to achieve target stinger tip clearance
    # model = ofx.Model(datpath)

    all_names = [obj.Name for obj in model.objects]
    roller_names = [x[3:] for x in all_names if x[:5] in ['b6 BR', 'b6 SR']]
    oroller = model['b6 ' + roller_names[-1]]

    line = model['Line1']
    vessel = [obj for obj in model.objects if obj.typeName == 'Vessel'][0]

    model.CalculateStatics()

    if target_var == 'tip_clearance':
        clr_sup1 = oroller.StaticResult('Support contact clearance', ofx.oeSupport(1))
        clr_sup2 = oroller.StaticResult('Support contact clearance', ofx.oeSupport(2))
        res0 = (clr_sup1 + clr_sup2) / 2
    else:
        res0 = line.StaticResult('Effective tension', ofx.oeEndA)

    bp0 = -vessel.StaticResult('Connections GX force')
    if bp0 < 10:
        bp0 = 10
    bp1 = bp0 * 1.05
    vessel.IncludedInStatics = '3 DOF'

    for _ in range(5):
        vessel.GlobalAppliedForceX[0] = bp1
        # Solve statics, use calculated line shapes
        solved = False
        for _ in range(5):
            try:
                model.CalculateStatics()
                model.UseCalculatedPositions(SetLinesToUserSpecifiedStartingShape=True)
                solved = True
                break
            except:
                newdamp = line.FullStaticsMinDamping + 2.0
                line.FullStaticsMinDamping = newdamp
                print(f'Increasing line damping to {newdamp}')

        if not solved:
            return False

        model.CalculateStatics()

        if target_var == 'tip_clearance':
            clr_sup1 = oroller.StaticResult('Support contact clearance', ofx.oeSupport(1))
            clr_sup2 = oroller.StaticResult('Support contact clearance', ofx.oeSupport(2))
            res1 = (clr_sup1 + clr_sup2) / 2
        else:
            res1 = line.StaticResult('Effective tension', ofx.oeEndA)

        if abs(res1 - target_val) < tol:
            break

        bp2 = (target_val - res0) / (res1 - res0) * (bp1 - bp0) + bp0
        # New BP limited to double old BP to avoid too large jump
        bp2 = min([float(x) for x in [bp1 * 2, bp2]])
        bp0, bp1 = bp1, bp2
        res0 = res1

    model.UseCalculatedPositions(SetLinesToUserSpecifiedStartingShape=True)

    return True


def get_setup_results(model: ofx.Model, datpath: Path) -> StingerSetupResults:

    if model.state == 0:
        # Reset state = 0, InStaticState = 2
        model.CalculateStatics()

    all_names = [obj.Name for obj in model.objects]
    roller_names = [x[3:] for x in all_names if x[:5] in ['b6 BR', 'b6 SR']]
    oroller = model['b6 ' + roller_names[-1]]

    line = model['Line1']
    vessel = [obj for obj in model.objects if obj.typeName == 'Vessel'][0]

    tip_depth, tip_angle, draft = stinger_tip_data(model)

    code_checks = model['Code checks']
    code_checks.DNVSTF101GammaF = 1.2
    code_checks.DNVSTF101GammaE = 0.7

    top_tension = float(line.StaticResult('Effective tension', ofx.oeEndA))

    last_roller_arc = float(oroller.tags['arc']) / 1000
    line_length = line.CumulativeLength[-1]
    arc_ob = ofx.arSpecifiedArclengths(0, last_roller_arc)
    arc_sag = ofx.arSpecifiedArclengths(last_roller_arc, line_length)

    var = 'DNV OS F101 load controlled'
    code_checks.DNVSTF101GammaC = 0.8
    uc_ob = line.RangeGraph(var, None, None, arc_ob).Mean.max()
    code_checks.DNVSTF101GammaC = 1.0
    uc_sag = line.RangeGraph(var, None, None, arc_sag).Mean.max()

    return StingerSetupResults(datpath, top_tension, uc_ob, uc_sag,
                               tip_depth, tip_angle, draft)


def remove_stinger_section(model: ofx.Model, isect: int):
    ''' Removes stinger section and associated constraints and rollers'''
    all_names = [obj.Name for obj in model.objects]

    constraints = [obj for obj in model.objects if obj.typeName == 'Constraint']
    for cn in constraints:
        if cn.Name[:5] == 'cn SR':
            if cn.InFrameConnection == f'b6 stinger_{isect}':
                buoy = cn.Name.replace('cn', 'b6')
                model.DestroyObject(cn)
                model.DestroyObject(buoy)

    model.DestroyObject(model[f'b6 stinger_{isect}'])

    return model


def get_base_case(vessel: Vessel, stinger_radius: float,
                  num_section: int = 3) -> ofx.Model:
    # - Add pipe linetype to vessel model,
    # - Delete stinger sections,
    # - update stinger_ref tags (radius, num_section, path length)

    datpath = Path(str(files('pypelay') / f'{vessel.name}.dat'))
    model = ofx.Model(datpath)

    # Fetch pipe linetype from pipe dat file
    datpath = PATH / 'pipe.dat'
    pmodel = ofx.Model(datpath)
    linetypes = [obj.Name for obj in pmodel.objects if obj.typeName == 'Line type']
    linetype = pmodel[linetypes[0]]
    clone = linetype.CreateClone(name=linetype.Name, model=model)

    # Remove stinger sections
    if num_section < 3:
        model = remove_stinger_section(model, 3)
    if num_section < 2:
        model = remove_stinger_section(model, 2)

    # Update stinger_ref tags
    stinger_ref = model['b6 stinger_ref']
    stinger_ref.tags['radius'] = f'{stinger_radius:.0f}'
    stinger_ref.tags['num_section'] = f'{num_section}'
    pthlen = stinger_ref.tags['path_length'].split('\n')
    pthlen = pthlen[num_section - 1].split(':')[1].strip()
    stinger_ref.tags['path_length'] = pthlen

    datpath = PATH / 'base case.dat'
    model.SaveData(datpath)

    return model


def get_roller_heights(
    model: ofx.Model, path_coords: np.ndarray,
    ang1: float, ang2: float) -> list[Roller]:
    ''' Calculates roller heights and angles for input stinger angles and pipe path.
        If pipe path is outside roller range (ymin, ymax) then returns None.'''

    all_names = [obj.Name for obj in model.objects]

    # Set stinger angles
    model['b6 stinger_1'].InitialRotation3 = ang1
    if 'b6 stinger_2' in all_names:
        model['b6 stinger_2'].InitialRotation3 = ang2

    ltypes = [obj for obj in model.objects if obj.typeName == 'Line type']
    pipe_od = ltypes[0].OD

    roller_names = [x[3:] for x in all_names if x[:5] in ['b6 BR', 'b6 SR']]
    nroller = len(roller_names)

    rollers = []
    for iroller, rname in enumerate(roller_names):
        # Get roller data from Orcaflex dat file
        buoy = model[f'b6 {rname}']
        cn = model[f'cn {rname}']
        # roller_conn = cn.InFrameConnection
        cn.InitialY = 0.0
        cn.DOFInitialValue[5] = 0.0
        buoy.Connection = 'b6 stinger_ref'

        roller_data = Roller(rname)
        origin_x = buoy.InitialX * 1000
        origin_y = buoy.InitialY * 1000
        post_angle = buoy.InitialRotation3 + 90
        ymin = float(buoy.tags['ymin']) * 1000
        ymax = float(buoy.tags['ymax']) * 1000
        yincr = float(buoy.tags['yincr']) * 1000
        length = float(buoy.tags['length']) * 1000
        v_angle = float(buoy.tags['v_angle'])
        # Very important: reset Orcaflex model roller
        buoy.Connection = f'cn {rname}'

        # Reduce max height for last roller to allow space for vertical rollers
        tip_capture = 2100
        if iroller == nroller - 1:
            ymax += -tip_capture

        roller_coords = np.array([origin_x, origin_y], dtype=float)
        # Get pipe path coords relative to roller_coords
        pipe_path2 = path_coords - roller_coords
        # Rotate pipe path so that support post is Y-axis
        c = math.cos(math.radians(post_angle - 90))
        s = math.sin(math.radians(post_angle - 90))
        xdash = pipe_path2[:, 0] * c + pipe_path2[:, 1] * s
        ydash = -pipe_path2[:, 0] * s + pipe_path2[:, 1] * c

        # Get pipe slope and curvature
        ind = np.argmin(np.abs(xdash))
        if xdash[ind] > 0:
            dx, dy = path_coords[ind + 1, :] - path_coords[ind, :]
        else:
            dx, dy = path_coords[ind, :] - path_coords[ind - 1, :]
        slope = math.degrees(math.atan2(dy, dx))
        roller_data.post_angle = post_angle
        roller_data.r3 = slope - (post_angle - 90)

        y_offset = (pipe_od/2) / math.cos(math.radians(v_angle))
        # Offset due to pipe curvature and distance between rollers
        curv = menger_curv(path_coords[ind - 1:ind + 2, :])
        if curv > 0:
            R = 1 / curv  # Pipe bend radius at roller
            y_offset += R * (1 - math.cos(length / (2 * R)))

        roller_data.y_offset = y_offset
        y_intersect = np.interp(0, xdash, ydash)
        y_roller = y_intersect - y_offset

        # Calculate pipe_path arc length at roller
        ind = xdash > 0
        xdash = np.insert(xdash[ind], 0, 0.0)
        ydash = np.insert(ydash[ind], 0, y_intersect)

        segdists = np.hypot(np.diff(xdash), np.diff(ydash))
        roller_data.arc = segdists.sum()

        if ymin <= y_roller <= ymax:
            # Adjust roller height to nearest pin hole
            nhole = round((ymax - ymin) / yincr + 1)
            y_hole = np.linspace(ymin, ymax, nhole)
            ind = np.argmin(np.abs(y_hole - y_roller))
            y_roller = y_hole[ind]
            roller_data.y = y_roller
        else:
            return []

        rollers.append(roller_data)

    return rollers


def main():
    pass


if __name__ == "__main__":
    main()
