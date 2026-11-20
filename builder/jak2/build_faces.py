"""Attach native Jak II gameplay morph channels and live eye surfaces."""
import argparse,json,os,sys
from pathlib import Path
import numpy as np


def packed_primitive_rows(raw, name, primitives, accessor, indices, part):
    from fr3_models import DRAW, model

    primitive = primitives[part]
    draws = sorted((draw for effect in model(raw, name)['effects'] for draw in effect['draws']),
                   key=lambda draw: DRAW.unpack(draw)[3])
    assert len(draws) == len(primitives)
    _, _, _, start, count, _, _ = DRAW.unpack(draws[part])
    packed_ids = indices[start:start+count]
    local_ids = accessor(primitive['indices']).ravel()
    offset = int(packed_ids[0]) - int(local_ids[0])
    assert np.array_equal(packed_ids, local_ids+offset)
    return np.unique(packed_ids), offset


def fit_jak_crown(raw, name, doc, accessor, indices, vertices):
    """Match the native crown travel during gameplay hair motion."""
    primitives = doc['meshes'][0]['primitives']
    rows,_ = packed_primitive_rows(raw, name, primitives, accessor, indices, 3)
    joint_names = [doc['nodes'][node]['name'] for node in doc['skins'][0]['joints']]
    # MERC vertex matrix IDs are one-based.
    neck = joint_names.index('neckB')+1
    hair = joint_names.index('MhairA')+1
    packed = np.frombuffer(vertices, dtype='<f4').reshape(-1, 16)
    matrices = np.frombuffer(vertices, dtype='u1').reshape(-1, 64)[:, 60:63]
    weights = packed[rows, 8:11]
    has_hair = ((matrices[rows] == hair) & (weights > 0)).any(1)
    rows = rows[(packed[rows, 1] > 2.64*4096) & has_hair]
    weights = packed[rows, 8:11]
    donor = (matrices[rows] == hair) & (weights > 0)
    anchor = (matrices[rows] == neck) & (weights > 0)
    assert np.all(donor.sum(1) == 1) and np.all(anchor.sum(1) == 1)
    # Keep the shape and the free ends of the hair. The top of the head
    # carries too much of the first hair joint's lift on the donor mesh.
    t = np.clip((packed[rows, 1]/4096-2.64)/.215, 0, 1)
    moved = .60*t*t*(3-2*t)*weights[donor]
    weights[donor] -= moved
    weights[anchor] += moved
    assert np.allclose(weights.sum(1), 1, atol=1e-6)
    packed[rows, 8:11] = weights
    return rows


def collar_sections(points, triangles, angles):
    """Measure the inner/outer radius and lower/upper hem around the neck."""
    edges = np.concatenate([triangles[:, pair] for pair in ((0, 1), (1, 2), (2, 0))])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    a, b = points[edges].transpose(1, 0, 2)
    envelope = []
    for angle in angles:
        radial = np.array([np.sin(angle), 0., np.cos(angle)])
        tangent = np.array([radial[2], 0., -radial[0]])
        da, db = a @ tangent, b @ tangent
        cut = (da*db <= 0) & (abs(da-db) > 1e-8)
        cross = a[cut] + (b[cut]-a[cut]) * (da[cut]/(da[cut]-db[cut]))[:, None]
        radius = cross @ radial
        radius, height = radius[radius > 0], cross[radius > 0, 1]
        envelope.append([radius.min(), radius.max(), height.min(), height.max()])
    return np.array(envelope)


def fit_jak_scarf(raw, name, doc, accessor, native, read_native, indices, vertices):
    """Wrap the donor's folds around the native collar and transfer its skinning."""
    from mesh_topology import components, nearest_triangle

    primitives = doc['meshes'][0]['primitives']
    scarf = primitives[17]
    stock = native['meshes'][0]['primitives'][16]
    assert 'scarf' in doc['materials'][scarf['material']]['name']
    assert 'scarf' in native['materials'][stock['material']]['name']
    rows, offset = packed_primitive_rows(raw, name, primitives, accessor, indices, 17)
    packed = np.frombuffer(vertices, dtype='<f4').reshape(-1, 16)
    matrices = np.frombuffer(vertices, dtype='u1').reshape(-1, 64)[:, 60:63]
    donor = accessor(scarf['attributes']['POSITION']).astype(float)
    original = read_native(stock['attributes']['POSITION']).astype(float)
    assert np.allclose(packed[rows, :3]/4096, donor[rows-offset], rtol=0, atol=.000004)

    def pieces(document, read, primitive, points):
        tri, groups = components(document, read, primitive)
        collar, *tabs = sorted((tri[g] for g in groups), key=len, reverse=True)
        assert len(tabs) == 2
        return [collar, *sorted(tabs, key=lambda t: points[t, 0].mean())]

    donor_pieces = pieces(doc, accessor, scarf, donor)
    stock_pieces = pieces(native, read_native, stock, original)
    angles = np.linspace(-np.pi, np.pi, 145)
    envelopes = [collar_sections(p, t[0], angles) for p, t in
                 ((donor, donor_pieces), (original, stock_pieces))]

    def wrap(points):
        x, y, z = points.T
        angle = np.arctan2(x, z)
        limits = [np.array([np.interp(angle, angles, v) for v in env.T]) for env in envelopes]
        inner, outer, bottom, top = limits[0]
        ri, ro, lo, hi = limits[1]
        radius = np.hypot(x, z)
        scale = (ri + (radius-inner)*(ro-ri)/(outer-inner)) / radius
        return np.column_stack((x*scale, lo + (y-bottom)*(hi-lo)/(top-bottom), z*scale))

    def rise(values, begin, end):
        t = np.clip((values-begin)/(end-begin), 0., 1.)
        return t*t*(3.-2.*t)

    def front_clearance(points):
        # Stock's shout rotates the torso in front of the longer cinematic
        # goatee. Seat the middle fold and let it partially follow the head.
        _, height, depth = points.T
        return rise(depth, .025, .16)*rise(height, 2.065, 2.165)*(1.-rise(height, 2.28, 2.32))

    jacket = primitives[11]
    assert 'jacketbody' in doc['materials'][jacket['material']]['name']
    coat = accessor(jacket['attributes']['POSITION'])[accessor(jacket['indices']).reshape(-1, 3)]
    a, b, c = coat.transpose(1, 0, 2)
    ab, ac = b-a, c-a
    det = ab[:, 0]*ac[:, 1] - ab[:, 1]*ac[:, 0]
    valid = abs(det) > 1e-10
    det = np.where(valid, det, 1.)
    def fit_hem(points):
        x, y, z = points.T
        # Follow the native collar up around the strap, with room for the chin
        # where it crosses the shoulder-side fold in the shout.
        toward_front = rise(z, .025, .16)
        fold = toward_front*(1.-rise(abs(x+.015), .15, .225))
        shoulder = rise(x, -.12, -.03)*(1.-rise(x, .18, .25))*rise(z, .03, .14)
        shape = points.copy()
        shape[:, 1] += .035*fold
        shape[:, 2] -= .024*shoulder

        # The cinematic jacket fills the opening underneath the stock front
        # hem. Fit cloth depth to its actual front surface so landing cannot
        # reveal an isolated piece of hem through the middle of the jacket.
        x, y, z = shape.T
        d = shape[:, None, :2] - a[None, :, :2]
        u = (d[:, :, 0]*ac[:, 1] - d[:, :, 1]*ac[:, 0])/det
        v = (ab[:, 0]*d[:, :, 1] - ab[:, 1]*d[:, :, 0])/det
        inside = valid & (u >= -1e-5) & (v >= -1e-5) & (u+v <= 1.00001)
        depth = np.where(inside, a[:, 2]+u*ab[:, 2]+v*ac[:, 2], -np.inf).max(1)
        lower_front = rise(z, .06, .135)*(1.-rise(y, 2.14, 2.21))
        shape[:, 2] += .8*np.maximum(depth+.050-z, 0.)*lower_front
        return shape

    def joint_names(document):
        return [document['nodes'][n]['name'] for n in document['skins'][0]['joints']]
    source_joints = joint_names(native)
    dest_joints = joint_names(doc)
    stock_weights = read_native(stock['attributes']['WEIGHTS_0'])
    stock_ids = read_native(stock['attributes']['JOINTS_0']).astype(int)
    stock_normals = read_native(stock['attributes']['NORMAL'])
    stock_colors = read_native(stock['attributes']['COLOR_0'])[:, :3]*255
    stock_uvs = read_native(stock['attributes']['TEXCOORD_0'])
    colors = np.frombuffer(vertices, dtype='u1').reshape(-1, 64)[:, 56:59]
    for piece, (source_tri, dest_tri) in enumerate(zip(stock_pieces, donor_pieces)):
        ids = np.unique(dest_tri)
        xyz = donor[ids]
        stock_xyz = original[source_tri]
        if piece == 0:
            fit_shape = wrap
        else:
            # Each separate tab has its own native anchor and free end.
            low, high = stock_xyz.min((0, 1)), stock_xyz.max((0, 1))
            base, span = xyz.min(0), np.ptp(xyz, axis=0)
            def fit_shape(points):
                return low + (points-base) * (high-low) / span

        reference = fit_shape(xyz)
        def transform(points):
            shape = fit_shape(points)
            shape[:, 2] -= .020 * front_clearance(shape)
            return fit_hem(shape) if piece == 0 else shape

        fitted = transform(xyz)
        rows_piece = ids + offset
        packed[rows_piece, :3] = fitted * 4096
        epsilon = 1e-5
        jacobian = np.stack([(transform(xyz+axis)-transform(xyz-axis))/(2*epsilon)
                             for axis in np.eye(3)*epsilon], axis=2)
        normals = np.linalg.solve(jacobian.transpose(0, 2, 1), packed[rows_piece, 4:7, None])[:, :, 0]
        packed[rows_piece, 4:7] = normals / np.linalg.norm(normals, axis=1, keepdims=True)

        # Interpolate on the matching cloth piece, so a tab cannot inherit a
        # shoulder joint from the nearby collar. MERC packs three influences.
        for row, point, clearance in zip(rows_piece, reference, front_clearance(reference)):
            index, bary, _ = nearest_triangle(point, stock_xyz)
            tri = source_tri[index]
            if piece == 0:
                # The front panel is exposed below the native rolled fold.
                # Carry its authored shade and continuous weave across the
                # donor's UV islands, while retaining the high-resolution atlas.
                x, y, z = point
                fade = rise(z, .050, .120)*(1.-rise(abs(x), .145, .220))*(1.-rise(y, 2.215, 2.285))
                if fade > 0:
                    target_normal = bary @ stock_normals[tri]
                    target_normal /= np.linalg.norm(target_normal)
                    n = (1.-fade)*packed[row, 4:7] + fade*target_normal
                    packed[row, 4:7] = n/np.linalg.norm(n)
                    colors[row] = np.rint((1.-fade)*colors[row] + fade*(bary @ stock_colors[tri]))
                    # Lift periodic coordinates onto the same chart before
                    # blending. Duplicates at the front and hem differ by one
                    # repeat; their physical fabric phase must stay continuous.
                    angle = np.arctan2(x, z)
                    bottom, top = (np.interp(angle, angles, col) for col in envelopes[1][:, 2:].T)
                    height = (y-bottom)/(top-bottom)
                    uv = packed[row, 12:14]
                    repeat = np.rint(uv - [1.-height, -2.*angle/np.pi])
                    uv += .20*fade*((bary @ stock_uvs[tri])+repeat-uv)
            weights = np.bincount(stock_ids[tri].ravel(),
                                  weights=(stock_weights[tri]*bary[:, None]).ravel(),
                                  minlength=len(source_joints))
            # Interpolate the stock field first, retaining the same three
            # influences MERC can store before seating the upper cloth.
            native_best = np.argsort(-weights, kind='stable')[:3]
            sparse = np.zeros_like(weights)
            sparse[native_best] = weights[native_best] / weights[native_best].sum()
            head_share = .35 * clearance
            weights = sparse * (1.-head_share)
            weights[source_joints.index('neckB')] += head_share
            best = np.argsort(-weights, kind='stable')[:3]
            packed[row, 8:11] = weights[best] / weights[best].sum()
            matrices[row] = [dest_joints.index(source_joints[j])+1 if weights[j] > 0 else 0 for j in best]
    return rows


def tuck_covered_neck(raw, name, doc, accessor, indices, vertices):
    """Seat the two front neck rings behind the cinematic collar."""
    prims = doc['meshes'][0]['primitives']
    face = prims[8]
    assert 'facert' in doc['materials'][face['material']]['name']
    rows, offset = packed_primitive_rows(raw, name, prims, accessor, indices, 8)
    pos_id, norm_id = (face['attributes'][k] for k in ('POSITION', 'NORMAL'))
    points = accessor(pos_id).copy()
    normals = accessor(norm_id).copy()
    ids = rows-offset
    def depth(xyz):
        x, y, z = xyz.T
        def rise(v, lo, hi):
            t = np.clip((v-lo)/(hi-lo), 0., 1.)
            return t*t*(3.-2.*t)
        # The chin lies farther forward, and the visible throat extends above
        # this band. The lower rings alone cross the shoulder-side cloth on idle.
        vertical = rise(y, 2.15, 2.18)*(1.-rise(y, 2.245, 2.280))
        front = rise(z, .080, .130)*(1.-rise(z, .157, .167))
        return .034*vertical*front*np.exp(-.5*((x-.035)/.070)**2)

    active = depth(points[ids]) > 1e-7
    ids, rows = ids[active], rows[active]
    xyz = points[ids].astype(float)
    # Transform the authored normals by the inverse transpose of the tuck.
    delta = np.eye(3)*1e-5
    grad = np.stack([(depth(xyz+d)-depth(xyz-d))/2e-5 for d in delta], axis=1)
    unit = normals[ids].astype(float)
    unit += grad*(unit[:, 2]/(1.-grad[:, 2]))[:, None]
    normals[ids] = unit/np.linalg.norm(unit, axis=1, keepdims=True)
    points[ids, 2] -= depth(xyz)
    packed = np.frombuffer(vertices, dtype='<f4').reshape(-1, 16)
    packed[rows, :3], packed[rows, 4:7] = points[ids]*4096, normals[ids]
    adjusted = {pos_id:points, norm_id:normals}
    def read(index):
        return adjusted[index] if index in adjusted else accessor(index)
    return rows, read


def fit_jak_clench(primitives, accessor, deformations):
    """Recover the native lower lip travel beside both canines."""
    clench, shout = 4, 8
    for part, (positions, _) in deformations.items():
        x, y, z = accessor(primitives[part]['attributes']['POSITION']).T
        # The shout separates the overlapping upper and lower lip rows. Use
        # that authored ownership and taper the correction into the jaw.
        lower_lip = np.clip((-positions[shout, :, 1] - .014) / .026, 0, 1)
        corners = np.exp(-((abs(x) - .048) / .026) ** 2 - ((y - 2.343) / .024) ** 2)
        corners *= np.clip((z - .248) / .036, 0, 1)
        corners *= np.clip((2.348 - y) / .012, 0, 1)
        motion = positions[clench, :, 1]
        motion += lower_lip * corners * np.minimum(0, -.0288 - motion)


def fit_jak_open_mouth(primitives, accessor, deformations):
    """Fit the small open mouth and keep both tooth rows inside the lips."""
    channel = 9
    skin = deformations[8][0]
    x, y, z = accessor(primitives[8]['attributes']['POSITION']).T
    # The fitted lower center travels 22 mm, against the original's 25 mm.
    # The shout distinguishes the lower lip from its overlapping upper row.
    lower = np.clip((-skin[8, :, 1] - .018) / .035, 0, 1)
    lip = np.exp(-((y - 2.341) / .030)**2 - (x / .067)**4)
    lip *= np.clip((z - .258) / .040, 0, 1)
    skin[channel, :, 1] -= .004 * lower * lip

    teeth = deformations[9][0]
    # The shout moves the two disconnected tooth arches in opposite
    # directions. The smaller opening needs the same ownership: the original
    # upper row rises while the lower teeth stay behind the descending lip.
    upper = teeth[8, :, 1] > 0
    teeth[channel, :, 1] += np.where(upper, .019, -.015)


def fit_jak_dental_expressions(primitives, accessor, deformations):
    """Fit the clench corners, broad smile, asymmetric smirk and open shout."""
    for part in (8, 9):
        p = primitives[part]
        vertices = accessor(p['attributes']['POSITION'])
        x, y, z = vertices.T
        positions, normals = deformations[part]
        # Keep the donor smile's continuous tooth band. Restrict the change
        # to the mouth so the separately fitted cheeks and brows keep moving.
        oral = np.clip((2.433-y)/.035, 0, 1) * np.clip((y-2.185)/.055, 0, 1)
        oral *= np.clip((z-.178)/.052, 0, 1)
        if part == 9:
            oral = np.ones_like(y)
        for kind, values in (('POSITION', positions), ('NORMAL', normals)):
            smile = accessor(p['targets'][1][kind])
            values[6] += oral[:, None] * (1.35*smile - values[6])
        px, py, _ = (vertices + positions[6]).T
        # The native smile reaches +/-79 mm and arcs upwards on both sides.
        # The donor's 55 mm half-width anchors the corner of the dental arch.
        corner = np.clip(abs(px)/.055, 0, 1)
        width = np.exp(-(px/.088)**4)
        positions[6, :, 0] += .44 * px * width * oral
        positions[6, :, 1] += oral * width * (
            .021*corner**2 - .005 - .30*np.clip(py-2.347, -.025, .025))

        # At peak clench the original lower edge drops towards the corners.
        # The older fit bends both dental edges upwards there. Expand only
        # the lower portion of the posed band, fading into the outer cheek.
        peak = 9856 / 8192
        px, py, _ = (vertices + peak*positions[4]).T
        lateral = np.clip(abs(px)/.052, 0, 1)
        cheek = np.exp(-(px/.087)**4) * oral
        below = np.clip((2.370-py)/.028, 0, 1)
        positions[4, :, 1] -= cheek * (.004 + .036*lateral**2*below) / peak
        positions[4, :, 0] += .10*px*cheek / peak

        # The native held frown keeps the two lip edges together. Its center
        # rises while the visible corners turn down and widen to +/-65 mm.
        # Move both closed donor lips and the hidden teeth with one field:
        # fitting either lip to the mixed smile leaves a gap at each corner.
        corner = np.clip((abs(x)-.021)/.034, 0, 1)
        corner_curve = corner**2 * (3-2*corner)
        shoulder = np.exp(-((abs(x)-.032)/.014)**2)
        lift = .024 + .0045*shoulder - .0075*corner_curve
        closed = np.column_stack((.20*x, lift, .002*(1-corner_curve)))
        frown = np.clip((.100-abs(x))/.032, 0, 1)
        frown *= np.clip((y-2.275)/.057, 0, 1) * np.clip((2.405-y)/.046, 0, 1)
        frown *= np.clip((z-.219)/.051, 0, 1)
        if part == 9:
            frown = np.ones_like(y)
        positions[5] += frown[:, None] * (closed-positions[5])
        # Transform the neutral crease normals by the local frown slope.
        base = accessor(p['attributes']['NORMAL'])
        slope = (6/.034) * corner*(1-corner) * np.sign(x)
        lift_slope = -.0075*slope - .009*(abs(x)-.032)*shoulder*np.sign(x)/.014**2
        tilted = base.copy()
        tilted[:, 0] = (base[:, 0] - lift_slope*base[:, 1] + .002*slope*base[:, 2])/1.20
        tilted /= np.maximum(np.linalg.norm(tilted, axis=1, keepdims=True), 1e-12)
        normals[5] += frown[:, None] * (tilted-base-normals[5])

        # The native positive smirk pulls the actor's positive-X corner
        # 8ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ15 mm higher than the fit. The negative input uses the same delta.
        raised_corner = np.exp(-((x-.060)/.037)**2 - ((y-2.344)/.032)**2)
        positions[7, :, 1] += .012 * raised_corner * oral

        # The native open shouts have a broader, flatter floor. Spread their
        # lip and dental corners together, fading into the fixed cheek.
        for channel, peak, spread in ((8, 6528/8192, .23), (10, 8512/8192, .14)):
            px, py, _ = (vertices + peak*positions[channel]).T
            opening = oral * np.exp(-(px/.091)**4)
            positions[channel, :, 0] += spread*px*opening / peak
            floor = np.clip((2.32-py)/.037, 0, 1)
            edges = np.exp(-((abs(px)-.044)/.024)**2)
            positions[channel, :, 1] -= .007 * opening * floor * edges / peak

    # In the open shout the native exposes a thin upper rim and no lower
    # band. Carry the lower teeth with the jaw, and lift the upper arch.
    tooth_motion = deformations[9][0]
    upper = tooth_motion[8, :, 1] > 0
    # At full clench the two dental arches meet; the fitted rows leave a
    # thin view through to the rear of the mouth.
    tooth_motion[4, :, 1] += np.where(upper, -.0015, .0025)
    tooth_motion[10, :, 1] += np.where(upper, .015, -.028)
    tooth_motion[7, :, 1] += np.where(upper, .006, -.003)

    # The high-detail tongue and soft palate share the skin primitive.
    # Seat the tongue with the jaw and retract the tip of the soft palate
    # into its arch at the two open-shout poses. All fields end behind the
    # lips, tapering into the interior attachment surfaces.
    x, y, z = accessor(primitives[8]['attributes']['POSITION']).T
    inside = np.clip((z-.150)/.044, 0, 1) * np.clip((.263-z)/.026, 0, 1)
    tongue = inside * np.exp(-(x/.053)**4)
    tongue *= np.clip((y-2.245)/.060, 0, 1) * np.clip((2.346-y)/.016, 0, 1)
    palate = inside * np.exp(-(x/.022)**4)
    palate *= np.clip((y-2.334)/.016, 0, 1) * np.clip((2.410-y)/.045, 0, 1)
    skin = deformations[8][0]
    for channel, tongue_drop, palate_lift in ((8, .041, .022), (10, .027, .014)):
        skin[channel, :, 1] += palate_lift*palate - tongue_drop*tongue
        skin[channel, :, 2] -= .009*palate + .010*tongue


def fit_jak_raised_eyes(lp, la, hp, ha, eye_parts, native_eyes, deformations, count):
    """Carry the raised-lid expressions across the whole cap and its upper rim."""
    from fix_faces import closest_surface
    for part, side in eye_parts.items():
        source, dest = lp[native_eyes[side]], hp[part]
        tri = la(source['indices']).reshape(-1, 3)
        reference = la(source['attributes']['POSITION'])[tri]
        used = np.unique(ha(dest['indices']))
        vertices = ha(dest['attributes']['POSITION'])
        cap = vertices[used]
        # The caps have different rest depths and widths. Locate matching
        # points within the individual cap before transferring its motion.
        native_min = reference.min(axis=(0, 1))
        native_span = reference.max(axis=(0, 1)) - native_min
        query = native_min + (cap-cap.min(axis=0)) * native_span / np.ptp(cap, axis=0)
        matches, bary, _ = closest_surface(query, reference)
        if part not in deformations:
            shape = (count, len(vertices), 3)
            deformations[part] = (np.zeros(shape), np.zeros(shape))
        for channel in (8, 9, 11):
            for kind, out in zip(('POSITION', 'NORMAL'), deformations[part]):
                delta = la(source['targets'][channel][kind])[tri[matches]]
                out[channel, used] = np.einsum('ni,nij->nj', bary, delta)

    # Across these three expressions the inner half of the native upper lid
    # rises 40ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ67 mm. The fit substitutes a round opening, leaving the inner
    # brow below the goggle frame. Move the skin and brow as a shared field.
    for part in (8, 10):
        x, y, z = ha(hp[part]['attributes']['POSITION']).T
        upper = np.clip((y-2.500)/.042, 0, 1) * np.clip((2.687-y)/.065, 0, 1)
        upper *= np.clip((z-.167)/.058, 0, 1)
        inner = upper * np.exp(-((abs(x)-.062)/.042)**2)
        positions = deformations[part][0]
        for channel, lift in ((8, .015), (9, .010), (11, .016)):
            positions[channel, :, 1] += lift*inner
        # The native smirk lifts the opposite upper lid while pinching the
        # eye on the smiling side; the fitted inner lid previously sank.
        positions[7, :, 1] += .010 * np.exp(-((x+.075)/.043)**2) * upper


def fit_daxter_shout_lids(lp, la, hp, ha, native_eyes, deformations):
    """Raise the fur opening and the underlying eye by the same native travel."""
    channel, peak = 8, 9792 / 8192
    travel = []
    for side in (1, 0):
        eye = lp[native_eyes[side]]
        used = np.unique(la(eye['indices']))
        y = la(eye['attributes']['POSITION'])[used, 1]
        dy = la(eye['targets'][channel]['POSITION'])[used, 1]
        travel.append((y + peak*dy).max() - y.max())
    for part, (positions, _) in deformations.items():
        posed = ha(hp[part]['attributes']['POSITION']) + peak*positions[channel]
        x, y, z = posed.T
        # The common vertical stretch preserves the existing 3D overlap
        # between the dense eye shell and the separately skinned fur rim.
        rise = np.interp(y, [1.078, 1.220, 1.295], [0, 1, 0])
        rise *= np.clip((z-.005)/.055, 0, 1)
        rise *= np.clip((abs(x)-.005)/.023, 0, 1)*np.clip((.170-abs(x))/.040, 0, 1)
        positions[channel, :, 1] += rise * np.interp(x, [-.030, .030], travel) / peak


def fit_daxter_poses(lp, la, hp, ha, low_eyes, deformations):
    from daxter_expression_fit import compress_closed_caps, fit_grin_tooth_rows, fit_mouth, fit_stretched_grimace, fit_tall_eyes

    # Values are the native comparison inputs in the gameplay weight units.
    poses = {
        0: (1., ((18, .65),), (1, 1, 1)),
        1: (1., ((0, 1.27),), (1.26, 1.37, 1)),
        2: (1., ((10, 1.),), (1, 1, 1)),
        3: (1., ((10, .58), (26, .78)), (1.06, .88, 1)),
        # Use the authored cinematic grin for the Dark Jak clench.
        4: (9664/8192, ((10, 1.),), (1.20, .95, 1)),
        5: (1., ((11, 1.26),), (1.1, 1.25, 1)),
        6: (8064/8192, ((10, .58), (25, .78)), (1.06, .88, 1)),
        7: (7552/8192, ((10, .86),), (1.04, .75, 1)),
        8: (9792/8192, ((10, .78),), (1.06, .70, 1)),
    }
    def donor_shape(part, target):
        authored = hp[part]['targets'][target]
        return ha(authored['POSITION']), ha(authored['NORMAL'])

    fit_mouth(hp, ha, deformations, donor_shape, poses, shout=1, rise_scale={4: .33})
    fit_grin_tooth_rows(hp[15], ha, deformations[15])
    fit_tall_eyes([lp[i] for i in low_eyes.values()], la, hp, ha, deformations,
                  ((1, 1.), (5, 1.), (6, 8064/8192), (8, 9792/8192)))
    fit_stretched_grimace([lp[i] for i in low_eyes.values()], la, hp, ha, deformations, 9792/8192)
    for part in (13, 14):
        # Follow the brow inward at the fully compressed diagnostic pose.
        deformations[part][0][0, :, 2] -= .038
    compress_closed_caps(hp, ha, deformations, channel=0, weight=1., fraction=.50)


def fit_shout_mouth(name, primitives, accessor, deformations):
    """Keep the dental rows and outer lip corners coupled to the native shout."""
    channel = 8
    if name == 'jakb-lod0':
        # The original upper teeth rise with the upper lip. The cinematic
        # fit instead lowers both rows, exposing a second band in the mouth.
        teeth, _ = deformations[9]
        motion = teeth[channel, :, 1]
        upper = motion > -.016
        motion[upper] += .022
        motion[~upper] -= .030
        peak = 6528 / 8192
        skin, _ = deformations[8]
        posed = accessor(primitives[8]['attributes']['POSITION']) + peak*skin[channel]
        x, y, z = posed.T
        # At the native shout both lower lip corners descend almost to the
        # center's height. The fitted cinematic corners stop above that line.
        corners = np.exp(-((abs(x)-.049)/.029)**2 - ((y-2.305)/.035)**2)
        corners *= np.clip((z-.19)/.055, 0, 1)
        skin[channel, :, 1] -= corners*.022/peak
        return
    # The source's dental half-width at the observed peak is 168 mm. The
    # cinematic fit reaches 115 mm. Expand the posed oral region together,
    # fading into the fixed skull above the cheeks and behind the muzzle.
    peak = 9792 / 8192
    for part, (positions, _) in deformations.items():
        posed = accessor(primitives[part]['attributes']['POSITION']) + peak*positions[channel]
        posed_x, y, z = posed.T
        # End below the lower eyelid so a wider grin cannot spread the socket.
        fade = np.clip((1.078-y)/.024, 0, 1) * np.clip((y-.901)/.058, 0, 1)
        fade *= np.clip((z-.01)/.060, 0, 1)
        # Preserve the outer cheek silhouette. The dental arcade widens
        # inside the muzzle, with progressively less movement beyond it.
        spread = np.interp(abs(posed_x), [0, .115, .195], [0, .053, 0])
        positions[channel, :, 0] += np.sign(posed_x) * spread * fade / peak


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True,type=Path)
    parser.add_argument('--work',required=True,type=Path)
    args=parser.parse_args()
    os.environ['CINEMATIC_BUILD_WORK']=str(args.work.resolve())
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'stages'))
    from fix_faces import closest_surface,fix_model,vec
    from inspect_faces import glb,buffers
    from fr3_models import decompress,compress,serialize,model,DRAW
    from eye_alignment import calibrated_eye_uvs,write_eye_uvs
    models=args.data/'decompiler_out/jak2/levels'
    raw=decompress(args.work/'GAME-materials.fr3')
    start,indices,_,original_vertices=buffers(raw,'jakb-lod0')
    vertices=bytearray(original_vertices);new_indices=indices;updates=[];reports=[]
    for name,high,transfer in [
        ('jakb-lod0','jak-highres-lod0',{3:[5],4:[6],8:[9,11],24:[9,11]}),
        ('daxter-lod0','daxter-highres-lod0',{}),
    ]:
        ld,la=glb(models/f'common/{name}.glb')
        hd,ha=glb(args.work/f'rig-prototypes/{name}.glb')
        lp,hp=ld['meshes'][0]['primitives'],hd['meshes'][0]['primitives']
        count=len(lp[0]['targets'])
        coefficients=np.load(args.work/f'face-refined/{name}-face-coefficients.npy')
        deformations={}
        for pi,p in enumerate(hp):
            dp=np.einsum('ct,tnj->cnj',coefficients,np.array([ha(t['POSITION']) for t in p['targets']]))
            dn=np.einsum('ct,tnj->cnj',coefficients,np.array([ha(t['NORMAL']) for t in p['targets']]))
            if pi in transfer:
                triangles=[];delta=[];normal=[]
                for si in transfer[pi]:
                    source=lp[si];ids=la(source['indices']).reshape(-1,3)
                    triangles.append(la(source['attributes']['POSITION'])[ids])
                    delta.append(np.array([la(t['POSITION'])[ids] for t in source['targets']]))
                    normal.append(np.array([la(t['NORMAL'])[ids] for t in source['targets']]))
                ti,bary,_=closest_surface(ha(p['attributes']['POSITION']),np.concatenate(triangles))
                # Jak II channels 0..3 control fingers and hair, independently of expression.
                dp[:4]=np.einsum('tnij,ni->tnj',np.concatenate(delta,axis=1)[:4,ti],bary)
                dn[:4]=np.einsum('tnij,ni->tnj',np.concatenate(normal,axis=1)[:4,ti],bary)
            if np.max(np.abs(dp))>1e-7:
                deformations[pi]=(dp,dn)
        if name == 'jakb-lod0':
            fit_jak_clench(hp, ha, deformations)
        def eyes(_):
            out={}
            for pi,p in enumerate(hp):
                mat=hd['materials'][p['material']]
                ti=mat.get('pbrMetallicRoughness',{}).get('baseColorTexture',{}).get('index')
                if ti is not None:
                    image=hd['images'][hd['textures'][ti]['source']]['name']
                    if image in ('programmer_eye_left','programmer_eye_right'):
                        out[pi]=int(image.endswith('right'))
            assert len(out)==2
            return out
        high_eyes=eyes(high)
        # Daxter's eye caps and eyelids deform together in the cinematic model.
        # A rigid cap can show through the brow when the fitted eyelid moves.
        if name != 'daxter-lod0':
            for eye in high_eyes:
                deformations.pop(eye,None)
        low_eyes={0:3,1:2} if name.startswith('jak') else {0:2,1:1}
        fit_shout_mouth(name, hp, ha, deformations)
        if name == 'daxter-lod0':
            fit_daxter_shout_lids(lp, la, hp, ha, low_eyes, deformations)
            fit_daxter_poses(lp, la, hp, ha, low_eyes, deformations)
        else:
            fit_jak_open_mouth(hp, ha, deformations)
            fit_jak_dental_expressions(hp, ha, deformations)
            fit_jak_raised_eyes(lp, la, hp, ha, high_eyes, low_eyes, deformations, count)
            assert 17 not in deformations
            scarf_rows=fit_jak_scarf(raw, name, hd, ha, ld, la, new_indices, vertices)
            neck_rows,ha=tuck_covered_neck(raw, name, hd, ha, new_indices, vertices)
            hair_rows=fit_jak_crown(raw, name, hd, ha, new_indices, vertices)
        def adapter(*_):return hd,ha,deformations,count,[]
        eye_uv,eye_report=calibrated_eye_uvs(lp,la,hp,ha,high_eyes,low_eyes,name)
        changed_eye_vertices=write_eye_uvs(raw,name,hd,ha,new_indices,vertices,eye_uv)
        native_record=model(decompress(args.data/'out/jak2/fr3/GAME.fr3'),name)
        native_eye_ids=sorted({DRAW.unpack(d)[2] for e in native_record['effects'] for d in e['draws'] if DRAW.unpack(d)[2]!=255})
        assert len(native_eye_ids)==2 and native_eye_ids[0]%2==0 and native_eye_ids[1]==native_eye_ids[0]+1
        target,added,report=fix_model(raw,name,high,{},new_indices,vertices,
                                     retarget_fn=adapter,eye_fn=eyes,
                                     eye_slot=native_eye_ids[0]//2)
        report['native_eye_ids']=native_eye_ids
        report['eye_alignment']=eye_report
        report['eye_alignment']['uvVerticesWritten']=len(changed_eye_vertices)
        target['tail']=target['tail'][:8]+native_record['tail'][8:]
        new_indices=np.concatenate([new_indices]+added);updates.append(target);reports.append(report)
    fixed=raw[:start]+vec(new_indices.astype('<u4').tobytes(),4)+vec(vertices,64)+raw[-2:]
    for target in sorted(updates,key=lambda x:x['start'],reverse=True):
        fixed=fixed[:target['start']]+serialize(target)+fixed[target['end']:]
    _,check_indices,_,check_vertices=buffers(fixed,'jakb-lod0')
    assert np.array_equal(check_indices[:len(indices)],indices)
    before=np.frombuffer(original_vertices,'u1').reshape(-1,64)
    after=np.frombuffer(check_vertices,'u1').reshape(-1,64)
    allowed=np.zeros(before.shape,dtype=bool)
    allowed[:,48:56]=True
    allowed[neck_rows,:12]=True
    allowed[neck_rows,16:28]=True
    allowed[scarf_rows,:12]=True
    allowed[scarf_rows,16:28]=True
    allowed[scarf_rows,32:44]=True
    allowed[scarf_rows,56:59]=True
    allowed[scarf_rows,60:63]=True
    allowed[hair_rows,32:44]=True
    assert np.array_equal(before[~allowed],after[~allowed])
    for target in updates:assert serialize(model(fixed,target['name']))==serialize(target)
    compress(args.work/'GAME-faces.fr3',fixed)
    (args.work/'native-faces-report.json').write_text(json.dumps(reports,indent=2)+'\n')
    print(json.dumps(reports,indent=2))

if __name__=='__main__':main()
