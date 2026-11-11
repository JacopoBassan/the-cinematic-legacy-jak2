"""Connected triangle components for local cinematic mesh fitting."""
import numpy as np


def components(doc, accessor, primitive):
    """Join authored coincident seams using the original five-decimal rule."""
    indices = accessor(primitive['indices']).ravel().reshape(-1, 3)
    points = accessor(primitive['attributes']['POSITION'])
    parents = np.arange(len(indices))
    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index
    seen = {}
    for triangle_index, triangle in enumerate(indices):
        for vertex in triangle:
            key = tuple(np.round(points[vertex], 5))
            if key in seen:
                parents[root(triangle_index)] = root(seen[key])
            else:
                seen[key] = triangle_index
    result = {}
    for index in range(len(indices)):
        result.setdefault(root(index), []).append(index)
    return indices, [np.asarray(part, dtype=int) for part in result.values()]


def nearest_triangle(point, triangles):
    """Exact closest point, considering interiors and every boundary edge."""
    a, b, c = triangles.transpose(1, 0, 2)
    ab, ac, ap = b-a, c-a, point-a
    d00, d01, d11 = (ab*ab).sum(1), (ab*ac).sum(1), (ac*ac).sum(1)
    d20, d21 = (ap*ab).sum(1), (ap*ac).sum(1)
    denominator = d00*d11-d01*d01
    valid = denominator > 1e-16
    v = np.divide(d11*d20-d01*d21, denominator, out=np.zeros_like(d00), where=valid)
    w = np.divide(d00*d21-d01*d20, denominator, out=np.zeros_like(d00), where=valid)
    bary = np.column_stack([1-v-w, v, w])
    distance = np.where(valid & (bary >= 0).all(1),
                        ((np.einsum('ni,nij->nj', bary, triangles)-point)**2).sum(1), np.inf)
    all_bary, all_distance = [bary], [distance]
    for i, j in ((0, 1), (1, 2), (2, 0)):
        edge = triangles[:, j]-triangles[:, i]
        length = (edge*edge).sum(1)
        t = np.clip(np.divide(((point-triangles[:, i])*edge).sum(1), length,
                              out=np.zeros_like(length), where=length > 1e-16), 0, 1)
        weights = np.zeros_like(bary)
        weights[:, i], weights[:, j] = 1-t, t
        all_bary.append(weights)
        all_distance.append(((triangles[:, i]+t[:, None]*edge-point)**2).sum(1))
    choices = np.asarray(all_distance)
    case, index = np.unravel_index(choices.argmin(), choices.shape)
    return int(index), all_bary[case][index], float(np.sqrt(choices[case, index]))
