"""层级骨骼、姿态与前向运动学（真实骨骼求值，非状态名切视频）。"""

from .math3d import qidentity, qmul, qnormalize, qrotate, qslerp, vadd, vlerp, vscale


class Bone:
    def __init__(self, name, parent, rest_dir, length):
        self.name = name
        self.parent = parent          # 父骨骼索引，根为 -1
        self.rest_dir = tuple(rest_dir)
        self.length = float(length)

    def to_dict(self):
        return {"name": self.name, "parent": self.parent,
                "rest_dir": list(self.rest_dir), "length": self.length}

    @classmethod
    def from_dict(cls, d):
        return cls(d["name"], d["parent"], d["rest_dir"], d["length"])


class Skeleton:
    def __init__(self, bones):
        self.bones = list(bones)
        self.index = {b.name: i for i, b in enumerate(self.bones)}

    @property
    def names(self):
        return [b.name for b in self.bones]

    def to_dict(self):
        return {"bones": [b.to_dict() for b in self.bones]}

    @classmethod
    def from_dict(cls, d):
        return cls([Bone.from_dict(b) for b in d["bones"]])


class Pose:
    """某一时刻的姿态：根位置 + 每根骨骼的局部旋转（相对父级）。"""

    def __init__(self, root_pos, rotations=None):
        self.root_pos = tuple(root_pos)
        self.rotations = dict(rotations or {})

    def local_rot(self, name):
        return self.rotations.get(name, qidentity())

    def to_dict(self):
        return {"root_pos": list(self.root_pos),
                "rotations": {k: list(v) for k, v in self.rotations.items()}}

    @classmethod
    def from_dict(cls, d):
        return cls(d["root_pos"], {k: tuple(v) for k, v in d["rotations"].items()})


def identity_pose(skeleton):
    return Pose((0.0, 0.0, 0.0), {b.name: qidentity() for b in skeleton.bones})


def fk(skeleton, pose):
    """前向运动学：返回每根骨骼的世界位置与世界旋转。"""
    positions = [None] * len(skeleton.bones)
    world_rots = [None] * len(skeleton.bones)
    for i, bone in enumerate(skeleton.bones):
        local = pose.local_rot(bone.name)
        if bone.parent < 0:
            world_rots[i] = qnormalize(local)
            positions[i] = tuple(pose.root_pos)
        else:
            wr = qnormalize(qmul(world_rots[bone.parent], local))
            world_rots[i] = wr
            offset = qrotate(wr, vscale(bone.rest_dir, bone.length))
            positions[i] = vadd(positions[bone.parent], offset)
    return positions, world_rots


def blend_pose(skeleton, pose_a, pose_b, weight_b):
    """按权重混合两个姿态；旋转走最近路径 slerp，根位置线性插值。"""
    t = max(0.0, min(1.0, weight_b))
    root = vlerp(pose_a.root_pos, pose_b.root_pos, t)
    names = set(pose_a.rotations) | set(pose_b.rotations)
    rots = {}
    for name in names:
        rots[name] = qslerp(pose_a.local_rot(name), pose_b.local_rot(name), t)
    return Pose(root, rots)
