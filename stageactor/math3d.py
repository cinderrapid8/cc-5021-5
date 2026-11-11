"""四元数/向量数学（仅用标准库，保证跨平台确定性）。

四元数约定为 (x, y, z, w)，即 w 为标量分量。
"""

import math


def vadd(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vsub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vscale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def vdot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vlen(a):
    return math.sqrt(vdot(a, a))


def vnorm(a):
    n = vlen(a)
    return (a[0] / n, a[1] / n, a[2] / n) if n > 1e-12 else (0.0, 0.0, 0.0)


def vcross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def vlerp(a, b, t):
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    )


def qidentity():
    return (0.0, 0.0, 0.0, 1.0)


def qnormalize(q):
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n) if n > 1e-12 else qidentity()


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def qconj(q):
    return (-q[0], -q[1], -q[2], q[3])


def qfrom_axis_angle(axis, angle):
    if abs(angle) < 1e-12:
        return qidentity()
    x, y, z = vnorm(axis)
    h = angle * 0.5
    s = math.sin(h)
    return (x * s, y * s, z * s, math.cos(h))


def qrotate(q, v):
    x, y, z, w = q
    # r = q * v * q^-1，展开形式
    u = (x, y, z)
    t = vscale(vcross(u, v), 2.0)
    return vadd(v, vadd(vscale(t, w), vcross(u, t)))


def qslerp(q1, q2, t):
    """球面线性插值；输入会先做最近路径对齐。"""
    q1 = qnormalize(q1)
    q2 = qnormalize(q2)
    d = q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3]
    if d < 0.0:
        q2 = (-q2[0], -q2[1], -q2[2], -q2[3])
        d = -d
    if d > 0.9995:
        r = (
            q1[0] + (q2[0] - q1[0]) * t,
            q1[1] + (q2[1] - q1[1]) * t,
            q1[2] + (q2[2] - q1[2]) * t,
            q1[3] + (q2[3] - q1[3]) * t,
        )
        return qnormalize(r)
    theta0 = math.acos(max(-1.0, min(1.0, d)))
    theta = theta0 * t
    sin_theta0 = math.sin(theta0)
    s1 = math.cos(theta) - d * math.sin(theta) / sin_theta0
    s2 = math.sin(theta) / sin_theta0
    return (
        s1 * q1[0] + s2 * q2[0],
        s1 * q1[1] + s2 * q2[1],
        s1 * q1[2] + s2 * q2[2],
        s1 * q1[3] + s2 * q2[3],
    )
