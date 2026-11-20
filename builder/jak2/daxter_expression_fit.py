"""Fit cinematic Daxter expressions to the gameplay mouth and eye controls."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def smoothstep(value):
    t = np.clip(value, 0, 1)
    return t*t*(3-2*t)


def lower_tooth_row(primitive, accessor, points):
    indices = accessor(primitive['indices'])
    used, triangles = np.unique(indices, return_inverse=True)
    welded, vertices = np.unique(points[used].round(6), axis=0, return_inverse=True)
    a, b, c = vertices[triangles.reshape(-1, 3)].T
    starts = np.concatenate((a, b, c))
    ends = np.concatenate((b, c, a))
    graph = coo_matrix((np.ones(len(starts)), (starts, ends)), shape=(len(welded),)*2)
    count, rows = connected_components(graph, directed=False)
    assert count == 2, 'Unexpected Daxter dental arch topology'
    lower = min(range(2), key=lambda row: welded[rows==row, 1].mean())
    return used[rows[vertices]==lower]


def fit_mouth(primitives, accessor, deformations, donor_shape, poses, shout, *, rise_scale=None):
    """Keep cohesive donor lip and dental rows at the native expression weights."""
    for part in (4, 5, 6, 15):
        if part not in deformations:
            continue
        primitive = primitives[part]
        points = accessor(primitive['attributes']['POSITION'])
        mouth = np.ones(len(points)) if part == 15 else smoothstep((1.090-points[:, 1])/.022)
        positions, normals = deformations[part]
        lower_row = lower_tooth_row(primitive, accessor, points) if part == 15 else None
        for channel, (weight, shapes, scale) in poses.items():
            position_delta = np.zeros_like(points)
            normal_delta = np.zeros_like(points)
            for target, influence in shapes:
                position, normal = donor_shape(part, target)
                position_delta += influence*position
                normal_delta += influence*normal
            position_delta *= scale
            if rise_scale and channel in rise_scale:
                # Retain the shape of the cinematic lip lift at the height
                # reached by the gameplay clench.
                positive_y = np.maximum(position_delta[:, 1], 0)
                position_delta[:, 1] -= (1-rise_scale[channel])*positive_y
            if part == 15 and channel == shout:
                # The donor shout only drops the lower row. The gameplay
                # shout also tucks it behind the lower lip and leaves no
                # upper tooth band crossing the opening.
                lower = np.zeros(len(points), dtype=bool)
                lower[lower_row] = True
                position_delta[lower, 1] -= .038
                position_delta[~lower, 1] += .010
                position_delta[~lower, 2] -= .021
                # Narrow the lower arch during the shout, where the sides
                # would otherwise poke through the descending jaw.
                position_delta[lower_row, 0] -= .35*(points[lower_row, 0]+position_delta[lower_row, 0])
                position_delta[lower_row, 2] -= .012
            positions[channel] += mouth[:, None]*(position_delta/weight-positions[channel])
            normals[channel] += mouth[:, None]*(normal_delta/weight-normals[channel])


def fit_tall_eyes(native_eyes, native_accessor, primitives, accessor, deformations, channels):
    """Stretch both the textured cap and its surrounding fur with the native lid."""
    def center(primitive, read):
        used = np.unique(read(primitive['indices']))
        return read(primitive['attributes']['POSITION'])[used, 0].mean()

    original = sorted(native_eyes, key=lambda p: center(p, native_accessor))
    cinematic = sorted((13, 14), key=lambda i: center(primitives[i], accessor))
    for side, cap_part, source in zip((-1, 1), cinematic, original):
        src_ids = np.unique(native_accessor(source['indices']))
        native_position = native_accessor(source['attributes']['POSITION'])[src_ids]
        y = native_position[:, 1]
        lower = y <= np.quantile(y, .25)
        upper = y >= np.quantile(y, .75)
        cap = primitives[cap_part]
        cap_ids = np.unique(accessor(cap['indices']))
        y_cap = accessor(cap['attributes']['POSITION'])[cap_ids, 1]
        bottom, height = y_cap.min(), np.ptp(y_cap)
        for channel, weight in channels:
            native_delta = native_accessor(source['targets'][channel]['POSITION'])[src_ids]
            posed = y + weight*native_delta[:, 1]
            upper_travel = posed.max()-y.max()
            if upper_travel < .025:
                # The opposite cap of the one-sided raise retracts farther
                # than its fur lid. Keep the donor fold and recover just the
                # missing cap height from the native compression ratio.
                if upper_travel < 0:
                    movement = deformations[cap_part][0][channel, :, 1]
                    posed_cap = y_cap + weight*movement[cap_ids]
                    missing = height*np.ptp(posed)/np.ptp(y) - np.ptp(posed_cap)
                    if missing > 0:
                        movement[cap_ids] += (posed_cap-posed_cap.min()) / np.ptp(posed_cap) * missing/weight
                continue
            lower_travel = posed.min()-y.min()
            lower_dx = native_delta[lower, 0].mean()
            upper_dx = native_delta[upper, 0].mean()
            for part in (4, 5, 6, cap_part):
                if part not in deformations:
                    continue
                x, y_pos, z = accessor(primitives[part]['attributes']['POSITION']).T
                fraction = np.clip((y_pos-bottom)/height, 0, 1)
                travel = (lower_travel + (upper_travel-lower_travel)*fraction)/weight
                lean = lower_dx + (upper_dx-lower_dx)*fraction
                movement = deformations[part][0][channel, :, 1]
                lateral = deformations[part][0][channel, :, 0]
                if part == cap_part:
                    movement[:] = travel
                    lateral[:] = lean
                    continue
                # Fade behind the head and across the nose. The tall side
                # of a one-sided reaction leaves its opposite squint intact.
                influence = smoothstep((y_pos-1.064)/.033)
                influence *= smoothstep(side*x/.037)*smoothstep((z-.006)/.040)
                movement += influence*(travel-movement)
                lateral += influence*(lean-lateral)


def compress_closed_caps(primitives, accessor, deformations, channel, weight, fraction):
    """Reduce the exposed white to the native squint without changing eye UVs."""
    for part in (13, 14):
        primitive = primitives[part]
        used = np.unique(accessor(primitive['indices']))
        motion = deformations[part][0][channel]
        y = accessor(primitive['attributes']['POSITION'])[used, 1] + weight*motion[used, 1]
        motion[used, 1] -= fraction*(y-y.mean())/weight

def cap_midpoints(points, indices):
    """Sample the center of the eye silhouette at matching vertical fractions."""
    triangles = points[indices.reshape(-1, 3)]
    edges = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    first, last = edges[:, 0], edges[:, 1]
    y = triangles[:, :, 1]
    levels = y.min() + np.array([.1, .5, .9]) * np.ptp(y)
    centers = []
    for level in levels:
        crosses = (first[:, 1] - level) * (last[:, 1] - level) < 0
        a, b = first[crosses], last[crosses]
        fraction = (level - a[:, 1]) / (b[:, 1] - a[:, 1])
        x = a[:, 0] + fraction * (b[:, 0] - a[:, 0])
        centers.append((x.min() + x.max()) / 2)
    return levels, np.array(centers)


def fit_stretched_grimace(native_eyes, native_accessor, primitives, accessor, deformations, peak):
    """Fit channel 8 to the stock sideways-flared grimace and inclined eye caps."""
    channel = 8
    # At the recorded peaks the native tooth band spans 327/323 mm. Its
    # lower edge at center sits 26/21 mm below the donor grin. The stock
    # upper edge is almost level across the muzzle in both games.
    jak2 = peak == 9792 / 8192
    for part in (4, 5, 6, 15):
        p = primitives[part]
        rest = accessor(p['attributes']['POSITION'])
        motion = deformations[part][0][channel]
        x, y, z = (rest + peak * motion).T
        radius = np.abs(x)
        mouth = smoothstep((1.106 - rest[:, 1]) / .039) * smoothstep((rest[:, 2] - .025) / .045)
        if part == 15:
            mouth = np.ones(len(x))
        width = (327 if jak2 else 323) / 278
        motion[:, 0] += mouth * x * (width - 1) / peak
        xs = [0, .035, .068, .090, .105, .120, .140]
        lower = np.interp(radius, xs, [-.026, -.020, -.013, -.003, .004, .004, -.010]
                          if jak2 else [-.021, -.016, -.010, 0, .006, .006, -.010])
        upper = np.interp(radius, xs, [-.008, -.008, -.010, -.010, -.012, -.016, -.019])
        # Raise the contact line relative to the newly opened lower lip.
        # The edge constraints apply to skin, lining and teeth in posed space.
        bottom = np.interp(radius, xs, [.992, .992, .993, .995, .999, 1.004, 1.022])
        top = np.interp(radius, xs, [1.041, 1.040, 1.041, 1.043, 1.045, 1.049, 1.050])
        level = np.clip((y - bottom) / (top - bottom), 0, 1)
        lift = .008 * (1 - np.abs(2 * level - 1))
        motion[:, 1] += mouth * (lower * (1 - level) + upper * level + lift) / peak

    def center(p, read):
        return read(p['attributes']['POSITION'])[np.unique(read(p['indices'])), 0].mean()

    originals = sorted(native_eyes, key=lambda p: center(p, native_accessor))
    caps = sorted((13, 14), key=lambda i: center(primitives[i], accessor))
    for side, cap, source in zip((-1, 1), caps, originals):
        native = native_accessor(source['attributes']['POSITION'])
        native = native + peak * native_accessor(source['targets'][channel]['POSITION'])
        _, desired = cap_midpoints(native, native_accessor(source['indices']))
        p = primitives[cap]
        current = accessor(p['attributes']['POSITION']) + peak * deformations[cap][0][channel]
        levels, fitted = cap_midpoints(current, accessor(p['indices']))
        for part in (4, 5, 6, cap):
            rest = accessor(primitives[part]['attributes']['POSITION'])
            motion = deformations[part][0][channel]
            y = rest[:, 1] + peak * motion[:, 1]
            influence = 1 if part == cap else (
                smoothstep((rest[:, 1] - 1.050) / .040) * smoothstep(side * rest[:, 0] / .030)
                * smoothstep((rest[:, 2] - .006) / .040))
            motion[:, 0] += influence * np.interp(y, levels, desired - fitted) / peak

def fit_grin_tooth_rows(primitive, accessor, deformations):
    """Match the native bite line while leaving the dental roots behind each lip."""
    points = accessor(primitive['attributes']['POSITION'])
    used = np.unique(accessor(primitive['indices']))
    lower = np.zeros(len(points), dtype=bool)
    lower[lower_tooth_row(primitive, accessor, points)] = True
    # The donor pairs each biting-edge vertex with its root at the same
    # depth and lateral coordinate. Use the same displacement at welded
    # seams and leave both sets of roots in their original positions.
    _, columns = np.unique(np.column_stack((lower[used], points[used, 0].round(4),
                                             points[used, 2].round(4))), axis=0, return_inverse=True)
    bottom = np.full(columns.max()+1, np.inf)
    top = np.full_like(bottom, -np.inf)
    y = points[used, 1]
    np.minimum.at(bottom, columns, y)
    np.maximum.at(top, columns, y)
    span = top[columns]-bottom[columns]
    assert np.all(span > .001), 'Unpaired cinematic dental column'
    fraction = (y-bottom[columns])/span
    edge = np.where(lower[used], fraction, 1-fraction)
    for channel, weight in ((4, 9664/8192), (6, 8064/8192), (7, 7552/8192)):
        motion = deformations[0][channel]
        posed_x = points[used, 0] + weight*motion[used, 0]
        if channel == 4:
            # Match the stock split within the authored clench aperture.
            lift = .009
        elif channel == 6:
            # The lip pulls up at one corner, but the stock bite line stays
            # nearly level across the middle and the raised side.
            lift = np.interp(posed_x, [-.125, -.105, -.085, -.060, -.035, 0, .035, .060, .085, .105, .125],
                             [0, .002, .007, .013, .013, .012, .012, .012, .012, .014, .015])
        else:
            # The native clench divides the band almost evenly across its width.
            lift = .0125
        motion[used, 1] += lift*edge/weight
        assert np.isfinite(motion[used]).all()
