"""Analytic 2-D geometry: oriented rectangles and circular pedestrians."""
import math
import numpy as np


def local(point, body):
    dx, dy = point[0] - body.x, point[1] - body.y
    c, s = math.cos(body.heading), math.sin(body.heading)
    return np.array([c * dx + s * dy, -s * dx + c * dy])


def corners(body):
    c, s = math.cos(body.heading), math.sin(body.heading)
    rotation = np.array([[c, -s], [s, c]])
    return np.array([[body.length / 2, body.width / 2],
                     [body.length / 2, -body.width / 2],
                     [-body.length / 2, -body.width / 2],
                     [-body.length / 2, body.width / 2]]) @ rotation.T + [body.x, body.y]


def segment_distance(p, a, b):
    ab = b - a
    u = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
    return float(np.linalg.norm(p - (a + u * ab)))


def clearance(a, b):
    """Nonnegative footprint clearance, zero means contact/overlap."""
    if a.kind == 'pedestrian' and b.kind == 'pedestrian':
        return max(0.0, math.hypot(a.x - b.x, a.y - b.y) - (a.width + b.width) / 2)
    if a.kind == 'pedestrian' or b.kind == 'pedestrian':
        ped, car = (a, b) if a.kind == 'pedestrian' else (b, a)
        q = np.abs(local((ped.x, ped.y), car)) - [car.length / 2, car.width / 2]
        return max(0.0, float(np.linalg.norm(np.maximum(q, 0))) - ped.width / 2)
    ca, cb = corners(a), corners(b)
    separated = False
    for body in (a, b):
        for angle in (body.heading, body.heading + math.pi / 2):
            axis = np.array([math.cos(angle), math.sin(angle)])
            pa, pb = ca @ axis, cb @ axis
            if pa.max() < pb.min() or pb.max() < pa.min():
                separated = True
    if not separated:
        return 0.0
    return min(segment_distance(p, poly[i], poly[(i + 1) % 4])
               for pts, poly in ((ca, cb), (cb, ca)) for p in pts for i in range(4))


def segment_blocked(start, end, blocker):
    """Open sight segment intersects blocker oriented rectangular footprint."""
    p, q = local(start, blocker), local(end, blocker)
    d = q - p
    low, high = 1e-6, 1 - 1e-6
    for j, bound in enumerate((blocker.length / 2, blocker.width / 2)):
        if abs(d[j]) < 1e-12:
            if abs(p[j]) > bound:
                return False
        else:
            t1, t2 = (-bound - p[j]) / d[j], (bound - p[j]) / d[j]
            low, high = max(low, min(t1, t2)), min(high, max(t1, t2))
            if low > high:
                return False
    return True
