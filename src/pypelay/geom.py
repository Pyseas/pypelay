
# from dataclasses import dataclass
import OrcFxAPI as ofx
import numpy as np
import math


def calc_path_coords(model: ofx.Model,
                     straight: float,
                     transition: float):
    # Dimensions in mm
    ltypes = [obj for obj in model.objects if obj.typeName == 'Line type']
    pipe_od = ltypes[0].OD * 1000

    stinger_ref = model['b6 stinger_ref']
    stinger_radius = float(stinger_ref.tags['radius'])
    endax = float(stinger_ref.tags['endax'])
    tensioner_x = float(stinger_ref.tags['tensioner_x'])
    bop = float(stinger_ref.tags['bop'])
    total_length = float(stinger_ref.tags['path_length'])

    pipe_y = bop + (pipe_od / 2) / math.cos(math.pi / 6)
    segment_length = 1000
    arc = total_length - (endax - tensioner_x) - straight - transition

    path_coords = [[endax, pipe_y], [tensioner_x, pipe_y]]
    if straight > 0:
        nseg = int(max(round(straight / segment_length), 2))
        seglen = straight / nseg
        xpos = tensioner_x
        for _ in range(nseg):
            xpos += -seglen
            path_coords.append([xpos, pipe_y])

    path_coords = np.array(path_coords, dtype=float)

    transition_coords = clothoid_coords(stinger_radius, transition, segment_length)
    transition_coords[:, 1:] *= -1  # Flip xy coords (curve is defined in positive xy quadrant)
    transition_coords[:, 1] += path_coords[-1, 0]  # Pipe path end point X coord, relative to FR0
    transition_coords[:, 2] += path_coords[-1, 1]

    path_coords = np.vstack((path_coords, transition_coords[1:, 1:]))

    trans_end_angle = transition / (2 * stinger_radius)
    ctr_angle = trans_end_angle  + 3/2 * math.pi
    arc_ctr = stinger_radius * np.array([math.cos(ctr_angle), math.sin(ctr_angle)])
    arc_ctr += transition_coords[-1, 1:]

    delta_angle = arc / stinger_radius
    start_angle = trans_end_angle + math.pi / 2
    end_angle = start_angle + delta_angle
    nseg = round(arc / segment_length)

    arc_coords = []
    for ang in np.linspace(start_angle, end_angle, nseg + 1):
        uvec = np.array([math.cos(ang), math.sin(ang)])
        arc_coords.append(arc_ctr + stinger_radius * uvec)

    arc_coords = np.array(arc_coords)
    path_coords = np.vstack((path_coords, arc_coords[1:, :]))
    path_coords = np.flipud(path_coords)

    return path_coords


def clothoid_coords(R: float, L: float, spc: float) -> np.ndarray:
    '''Calculate points along a clothoid (Euler spiral)

    Parameters
    ----------
    R : float
        Final bend radius.
    L : float
        Curve total length.
    spc: float
        Point spacing.

    Returns
    -------
    points: np.ndarray
        Num points x 3 array. First column is arc length.

    '''
    nseg = round(L / spc)
    pipe_arc = np.linspace(0, L, nseg + 1)
    x = pipe_arc - pipe_arc**5 / (40 * (R*L)**2)
    y = pipe_arc**3 / (6 * R * L) - pipe_arc**7 / (336 * (R*L)**3)

    return np.vstack((pipe_arc, x, y)).T


def menger_curv(points: np.ndarray) -> float:
    '''Calculates Menger curvature from 3 points (ref Wikipedia)

    Parameters
    ----------
    points : np.ndarray
        Numpy array with 3 rows, 2 columns (x, y)

    Returns
    -------
    curvature: float
        Curvature with min limit of 1E-6

    '''
    x = points[0, :]
    y = points[1, :]
    z = points[2, :]

    yx = x - y
    u_yx = yx / np.linalg.norm(yx)
    yz = z - y
    u_yz = yz / np.linalg.norm(yz)

    xz = np.linalg.norm(z - x)

    ang = math.acos(np.dot(u_yx, u_yz))

    curv = float(2 * math.sin(ang) / xz)
    # Set curvature to zero if radius > 1000m
    if curv < 1E-6:
        curv = 0

    return curv


def catenary_length(dh: float, top_angle: float) -> tuple[float, float]:
    # Arc length for a catenary from x=0, with height difference of dh
    a = dh / (1 / math.cos(top_angle) - 1)
    # y0 = dh + a
    dx = a * np.arcsinh(math.tan(top_angle))
    s0 = a * np.sinh(dx / a)  # Arc length

    return s0, dx
