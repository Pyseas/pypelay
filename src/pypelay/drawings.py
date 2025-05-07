
import math
import numpy as np
import ezdxf
from ezdxf.enums import TextEntityAlignment
from ezdxf.math import offset_vertices_2d
import OrcFxAPI as ofx
from pathlib import Path
from importlib.resources import files

__all__ = ['write_dxf']


def pin_hole(y: float, ymin: float, yincr: float) -> str:
    # Calculates roller pin hole (e.g. D-10) from roller height
    ihole = round((y - ymin) / yincr)
    roller_holes = ['A', 'B', 'C', 'D', 'E', 'F']
    rl_hole = roller_holes[ihole % 6]
    st_hole = ihole % 6 + int(ihole/6) + 1

    return f'{rl_hole}-{st_hole}'


def write_dxf_ga(datpath: Path) -> None:
    ''' Text is all ROMANS, height 680, layer DIM
        Update stinger roller settings table: stinger angle, BOP height,
        stinger hole, roller hole. Add labels for all stinger rollers
        (e.g. SR #1). Angular dimensions between stinger sections'''

    model = ofx.Model(datpath)

    stinger_ref = model['b6 stinger_ref']
    vessel_name = stinger_ref.Connection[2:]

    all_names = [obj.Name for obj in model.objects]

    dxfpath = Path(str(files('pypelay') / f'{vessel_name}_ga.dxf'))
    doc = ezdxf.readfile(dxfpath)
    msp = doc.modelspace()

    # Insert stinger sections
    for block_name in ['stinger_1', 'stinger_2', 'stinger_3']:
        if f'b6 {block_name}' in all_names:
            buoy = model[f'b6 {block_name}']
            buoy.Connection = 'b6 stinger_ref'
            glob_x = buoy.InitialX * 1000
            glob_y = buoy.InitialY * 1000
            r3 = buoy.InitialRotation3
            msp.add_blockref(
                block_name, (glob_x, glob_y),
                dxfattribs={'rotation': r3, 'layer': 'STINGER'})

    # Table top left corner (for stinger rollers)
    tbl_x, tbl_y = 35169.52, 143297.41
    row_ht, col_wd = 1860.22, 4662.38

    # Input stinger angle in first row
    text_x = tbl_x + 6 * col_wd
    text_y = tbl_y - 0.5 * row_ht
    ang1 = model['b6 stinger_1'].InitialRotation3
    msp.add_text(f'{ang1:.1f}', height=680,
        dxfattribs={'style': 'ROMANS', 'layer': 'DIM', 'color': 2}
        ).set_placement((text_x, text_y),
                        align=TextEntityAlignment.MIDDLE_CENTER)

    barge_rlr_names = [x[3:] for x in all_names if x[:5] == 'b6 BR']
    stinger_rlr_names = [x[3:] for x in all_names if x[:5] == 'b6 SR']
    rlr_names = barge_rlr_names + stinger_rlr_names

    # Insert roller blocks
    for rname in rlr_names:
        if rname[:1] == 'B':
            block_name = 'barge_roller'
        else:
            block_name = 'stinger_roller'
        buoy = model[f'b6 {rname}']
        buoy.Connection = 'b6 stinger_ref'
        glob_x = buoy.InitialX * 1000
        glob_y = buoy.InitialY * 1000
        r3 = buoy.InitialRotation3
        # Insert roller block
        msp.add_blockref(
            block_name, (glob_x, glob_y),
            dxfattribs={'rotation': r3, 'layer': 'ROLLER'})

    text_attribs = {'style': 'ROMANS', 'layer': 'DIM', 'color': 2}

    # Barge roller table text
    for irlr, rname in enumerate(barge_rlr_names):
        cn = model[f'cn {rname}']
        y = round(cn.InitialY * 1000)
        text_x = 29770 + irlr * 5860
        text_y = 107656
        msp.add_text(f'{y}', height=680,
            dxfattribs=text_attribs
            ).set_placement((text_x, text_y),
                            align=TextEntityAlignment.MIDDLE_CENTER)

    # Stinger roller labels and table text
    for irlr, rname in enumerate(stinger_rlr_names):
        # Label
        cn = model[f'cn {rname}']
        buoy = model[f'b6 {rname}']
        y = round(cn.InitialY * 1000)
        ymin = float(buoy.tags['ymin']) * 1000
        yincr = float(buoy.tags['yincr']) * 1000
        rl_hole, st_hole = pin_hole(y, ymin, yincr).split('-')
        buoy.Connection = cn.InFrameConnection
        buoy.InitialY = 0
        buoy.InitialRotation3 = 0
        buoy.Connection = 'b6 stinger_ref'
        origin_x = buoy.InitialX * 1000
        origin_y = buoy.InitialY * 1000
        post_angle = math.radians(buoy.InitialRotation3 - 90)
        label_offset = 800
        label_x = origin_x + label_offset * math.cos(post_angle)
        label_y = origin_y + label_offset * math.sin(post_angle)
        text = f'SR #{rname[2:]}'
        msp.add_text(text, height=680,
            dxfattribs=text_attribs
            ).set_placement((label_x, label_y),
                            align=TextEntityAlignment.TOP_LEFT)
        # Table text
        text_x = tbl_x + (irlr + 0.5) * col_wd
        text_y = tbl_y - 1.5 * row_ht
        for text in [f'{y}', st_hole, rl_hole]:
            msp.add_text(text, height=680,
                dxfattribs=text_attribs
                ).set_placement((text_x, text_y),
                                align=TextEntityAlignment.MIDDLE_CENTER)
            text_y -= row_ht

    # Add BOP profile
    # Reopen model since the roller positions in the first one are all messed up
    model = ofx.Model(datpath)
    line = model['Line1']
    stinger_ref = model['b6 stinger_ref']
    # glob_x, glob_y = line.EndAX * 1000, line.EndAY * 1000
    ltype = model[line.LineType[1]]
    pipe_od = ltype.OuterContactDiameter * 1000
    model.CalculateStatics()
    model.SaveSimulation(datpath.with_suffix('.sim'))
    ref_x = stinger_ref.StaticResult('X')
    ref_y = stinger_ref.StaticResult('Z')
    pipe_x = (line.RangeGraph('X').Mean - ref_x) * 1000
    pipe_y = (line.RangeGraph('Z').Mean - ref_y) * 1000
    pipe_cl = []
    for xpos, ypos in zip(pipe_x, pipe_y):
        pipe_cl.append([xpos, ypos])
    
    # msp.add_lwpolyline(pipe_cl)
    bop = list(offset_vertices_2d(pipe_cl, offset=pipe_od/2, closed=True))

    msp.add_lwpolyline(bop)

    outpath = datpath.parent / f'{datpath.stem}_ga.dxf'
    doc.saveas(outpath)


def write_dxf_rollers(datpath: Path) -> None:

    model = ofx.Model(datpath)

    stinger_ref = model['b6 stinger_ref']
    vessel_name = stinger_ref.Connection[2:]

    all_names = [obj.Name for obj in model.objects]
    roller_names = [x[3:] for x in all_names if x[:5] == 'b6 SR']
    nrlr = len(roller_names)

    dxfpath = Path(str(files('pypelay') / f'{vessel_name}_rollers.dxf'))

    doc = ezdxf.readfile(dxfpath)
    msp = doc.modelspace()

    base_points = np.zeros((nrlr, 2), dtype=float)
    for ref in msp.query('INSERT'):
        if ref.dxf.name[:8] == 'section_':
            irlr = int(ref.dxf.name[8:]) - 1
            if irlr < nrlr:
                x, y, _ = ref.dxf.insert
                base_points[irlr, :] = (x, y)

    rl_insert = [0] * nrlr
    st_insert = [0] * nrlr
    txt = ''
    for ref in msp.query('INSERT[name=="A3 - TAGG"]'):
        x, y, _ = ref.dxf.insert
        txt = ref.get_attrib('1').dxf.text
        for irlr in range(nrlr):
            xbase, ybase = base_points[irlr, :]
            if (0 < x - xbase < 7500) and (0 < y - ybase < 10000):
                if txt.isnumeric():
                    st_insert[irlr] = ref
                else:
                    rl_insert[irlr] = ref
                break

    leaders = [0] * nrlr
    for ref in msp.query('LEADER'):
        x, y, _ = ref.vertices[0]
        for irlr in range(nrlr):
            xbase, ybase = base_points[irlr, :]
            if (0 < x - xbase < 7500) and (0 < y - ybase < 10000):
                leaders[irlr] = ref

    for roller_name in roller_names:
        irlr = int(roller_name[2:]) - 1
        base_point = base_points[irlr]
        oroller = model['b6 ' + roller_name]
        ocn = model['cn ' + roller_name]
        ymin = float(oroller.tags['ymin'])
        yincr = float(oroller.tags['yincr'])
        y = ocn.InFrameInitialY
        # Insert roller
        rlr_x = base_point[0]
        rlr_y = base_point[1] + y * 1000
        msp.add_blockref('SR1_front', (rlr_x, rlr_y))
        # Update labels
        pin = pin_hole(y, ymin, yincr)
        rl_txt, st_txt = pin.split('-')
        attrib = rl_insert[irlr].get_attrib('1')
        attrib.dxf.text = rl_txt
        attrib = st_insert[irlr].get_attrib('1')
        attrib.dxf.text = st_txt
        # Update leader arrow to point at hole
        leader = leaders[irlr]
        st_hole = int(st_txt) - 1
        # Hard-coded values here ****
        y_hole = (ymin * 1000 - 646) + 180 * st_hole
        verts = leader.vertices
        arw_x, arw_y, _ = verts[0]
        verts[0] = (arw_x, base_point[1] + y_hole, 0)
        leader.set_vertices(verts)
        # Add value to table
        if irlr < 7:
            text_x, text_y = (29895.94, -43171.45)
            text_y += -800 * irlr
        else:
            text_x, text_y = (97273.88, -24851.87)
            text_y += -800 * (irlr - 7)
        msp.add_text(f'{y:.3f}', height=340,
            dxfattribs={'style': 'ROMANS', 'layer': 'DIM', 'color': 2}
            ).set_placement((text_x, text_y), align=TextEntityAlignment.MIDDLE_CENTER)

    outpath = datpath.parent / f'{datpath.stem}_rollers.dxf'
    doc.saveas(outpath)


def write_dxf(datpath: Path) -> None:
    """Create standard stinger configuration drawings in dxf format.

    Two dxf files are created: one showing the full stinger, the other
    including just the rollers.

    The new dxf file paths are modified versions of the input datpath.
    For example, an input of *R120.dat* will create *R120_ga.dxf* and
    *R120_rollers.dxf*.

    Args:
        datpath: File path of Orcaflex dat file
    """

    write_dxf_ga(datpath)

    write_dxf_rollers(datpath)
