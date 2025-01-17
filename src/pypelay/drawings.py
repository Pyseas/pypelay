
import math
import ezdxf
from ezdxf.enums import TextEntityAlignment
from ezdxf.math import offset_vertices_2d
import OrcFxAPI as ofx
from pathlib import Path
from importlib.resources import files


def pin_hole(y: float, ymin: float, yincr: float) -> str:
    # Calculates roller pin hole (e.g. D-10) from roller height
    ihole = round((y - ymin) / yincr)
    roller_holes = ['A', 'B', 'C', 'D', 'E', 'F']
    rl_hole = roller_holes[ihole % 6]
    st_hole = ihole % 6 + int(ihole/6) + 1

    return f'{rl_hole}-{st_hole}'


def write_dxf(datpath: Path) -> None:
    ''' Text is all ROMANS, height 680, layer DIM
        Update stinger roller settings table: stinger angle, BOP height,
        stinger hole, roller hole. Add labels for all stinger rollers (e.g. SR #1)
        Angular dimensions between stinger sections'''

    roller_names = ['BR5', 'BR6'] + [f'SR{i}' for i in range(1, 13)]
    model = ofx.Model(datpath)
    all_names = [obj.Name for obj in model.objects]

    dxfpath = Path(str(files('pipelay').joinpath('S1200_stcfg.dxf')))
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
    text_x = tbl_x + 6 * col_wd
    text_y = tbl_y - 0.5 * row_ht

    # Barge roller table text coords
    br_tbl_coords = {
        5: [26694.9, 107655.9],
        6: [32554.3, 107655.9],
    }

    # Input stinger angle in first row
    ang1 = model['b6 stinger_1'].InitialRotation3
    msp.add_text(f'{ang1:.1f}', height=680,
        dxfattribs={'style': 'ROMANS', 'layer': 'DIM', 'color': 2}
        ).set_placement((text_x, text_y), align=TextEntityAlignment.MIDDLE_CENTER)

    # Insert rollers
    for rname in roller_names:
        if f'b6 {rname}' not in all_names:
            continue
        BR = False
        if rname[:1] == 'B':
            BR = True
            block_name = 'barge_roller'
        else:
            block_name = 'stinger_roller'

        cn = model[f'cn {rname}']
        buoy = model[f'b6 {rname}']
        y = round(cn.InitialY * 1000)
        ymin = float(buoy.tags['ymin']) * 1000
        yincr = float(buoy.tags['yincr']) * 1000
        buoy.Connection = 'b6 stinger_ref'
        glob_x = buoy.InitialX * 1000
        glob_y = buoy.InitialY * 1000
        r3 = buoy.InitialRotation3

        # Insert roller
        msp.add_blockref(
            block_name, (glob_x, glob_y),
            dxfattribs={'rotation': r3, 'layer': 'ROLLER'})

        irlr = int(rname[2:])

        # Add text: table text and stinger roller labels
        if BR:
            text_x, text_y = br_tbl_coords[irlr]
            msp.add_text(f'{y}', height=680,
                dxfattribs={'style': 'ROMANS', 'layer': 'DIM', 'color': 2}
                ).set_placement((text_x, text_y), align=TextEntityAlignment.MIDDLE_CENTER)
        else:
            rl_hole, st_hole = pin_hole(y, ymin, yincr).split('-')
            # Add label under roller post
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
            text = f'SR #{irlr}'
            msp.add_text(text, height=680,
                dxfattribs={'style': 'ROMANS', 'layer': 'DIM', 'color': 2}
                ).set_placement((label_x, label_y), align=TextEntityAlignment.TOP_LEFT)

            # Add table text (stinger rollers)
            text_x = tbl_x + (irlr - 0.5) * col_wd
            text_y = tbl_y - 1.5 * row_ht
            for text in [f'{y}', st_hole, rl_hole]:
                msp.add_text(text, height=680,
                    dxfattribs={'style': 'ROMANS', 'layer': 'DIM', 'color': 2}
                    ).set_placement((text_x, text_y), align=TextEntityAlignment.MIDDLE_CENTER)
                text_y -= row_ht

    # Add BOP profile
    # Reopen model since the roller positions in the first one are all messed up
    model = ofx.Model(datpath)
    line = model['Line1']
    glob_x, glob_y = line.EndAX * 1000, line.EndAY * 1000
    ltype = model[line.LineType[0]]
    pipe_od = ltype.OD * 1000
    model.CalculateStatics()
    model.SaveSimulation(datpath.with_suffix('.sim'))
    pipe_x = line.RangeGraph('X').Mean * 1000
    pipe_y = line.RangeGraph('Z').Mean * 1000
    pipe_x += glob_x - pipe_x[0]
    pipe_y += glob_y - pipe_y[0]
    pipe_cl = []
    for xpos, ypos in zip(pipe_x, pipe_y):
        pipe_cl.append([xpos, ypos])
    
    bop = list(offset_vertices_2d(pipe_cl, offset=pipe_od/2, closed=True))
    # msp.add_lwpolyline(pipe_cl)
    msp.add_lwpolyline(bop)

    doc.saveas(datpath.with_suffix('.dxf'))
