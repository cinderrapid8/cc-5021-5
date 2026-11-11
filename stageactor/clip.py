"""动画片段：关键帧采样、循环尾部跨接插值、非循环结束保持、事件标记。

两套结束语义分别建模：
* 循环片段（loop=True）：相位对 duration 取模，尾部 tail_blend 窗口内做
  “本周期末姿态 -> 下一周期初姿态”的跨尾部插值；
* 非循环片段（loop=False）：采样时间被钳到 duration，结束后保持末姿态。
事件标记（脚步/提示）以显式时间点存储，由状态机按策略触发，不由采样器隐式产生。
"""

from .math3d import qslerp, vlerp
from .skeleton import Pose


class Keyframe:
    def __init__(self, time, pose):
        self.time = float(time)
        self.pose = pose

    def to_dict(self):
        return {"time": self.time, "pose": self.pose.to_dict()}

    @classmethod
    def from_dict(cls, d):
        return cls(d["time"], Pose.from_dict(d["pose"]))


class EventMarker:
    def __init__(self, time, name, data=None):
        self.time = float(time)
        self.name = str(name)
        self.data = dict(data or {})

    def to_dict(self):
        return {"time": self.time, "name": self.name, "data": self.data}

    @classmethod
    def from_dict(cls, d):
        return cls(d["time"], d["name"], d.get("data"))


class AnimationClip:
    def __init__(self, name, duration, keyframes, events=None, loop=False,
                 tail_blend=0.0):
        self.name = name
        self.duration = float(duration)
        self.keyframes = sorted(keyframes, key=lambda k: k.time)
        self.events = sorted(list(events or []), key=lambda e: e.time)
        self.loop = bool(loop)
        # 仅对循环片段有意义：跨尾部插值窗口长度
        self.tail_blend = float(tail_blend) if loop else 0.0

    # ------------------------------------------------------------------ 采样
    def _raw_cyclic(self, t):
        """在首尾关键帧构成的循环时间轴上采样，t 允许为负或超过 duration。"""
        keys = self.keyframes
        t0, tn = keys[0].time, keys[-1].time
        span = tn - t0
        u = (t - t0) % span          # 负时间回绕到上一周期
        return self._interp_at(t0 + u)

    def _interp_at(self, t):
        keys = self.keyframes
        if t <= keys[0].time:
            return Pose(keys[0].pose.root_pos, dict(keys[0].pose.rotations))
        if t >= keys[-1].time:
            return Pose(keys[-1].pose.root_pos, dict(keys[-1].pose.rotations))
        k, k1 = self._bracket(t)
        return self._interp(k, k1, t)

    def _bracket(self, t):
        keys = self.keyframes
        lo, hi = 0, len(keys) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if keys[mid].time <= t:
                lo = mid
            else:
                hi = mid
        return keys[lo], keys[hi]

    @staticmethod
    def _interp(k0, k1, t):
        span = k1.time - k0.time
        u = 0.0 if span <= 1e-12 else (t - k0.time) / span
        root = vlerp(k0.pose.root_pos, k1.pose.root_pos, u)
        rots = {}
        for name in set(k0.pose.rotations) | set(k1.pose.rotations):
            rots[name] = qslerp(k0.pose.local_rot(name),
                                k1.pose.local_rot(name), u)
        return Pose(root, rots)

    def sample(self, local_time):
        """返回 (姿态, 相位, 是否已结束)。非循环结束后保持末帧姿态。"""
        keys = self.keyframes
        if self.loop:
            phase = local_time % self.duration
            d = self.tail_blend
            if d > 0.0 and phase >= self.duration - d:
                u = phase - (self.duration - d)
                a = self._raw_cyclic(phase)
                b = self._raw_cyclic(phase - self.duration)  # 下一周期初姿态
                return self._blend_pose(a, b, u / d), phase, False
            return self._raw_cyclic(phase), phase, False

        if local_time >= self.duration:
            return Pose(keys[-1].pose.root_pos,
                        dict(keys[-1].pose.rotations)), self.duration, True
        k, k1 = self._bracket(local_time)
        return self._interp(k, k1, local_time), local_time, False

    @staticmethod
    def _blend_pose(a, b, t):
        root = vlerp(a.root_pos, b.root_pos, t)
        rots = {n: qslerp(a.local_rot(n), b.local_rot(n), t)
                for n in set(a.rotations) | set(b.rotations)}
        return Pose(root, rots)

    # --------------------------------------------------------------- 序列化
    def to_dict(self):
        return {
            "name": self.name, "duration": self.duration, "loop": self.loop,
            "tail_blend": self.tail_blend,
            "keyframes": [k.to_dict() for k in self.keyframes],
            "events": [e.to_dict() for e in self.events],
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            d["name"], d["duration"],
            [Keyframe.from_dict(k) for k in d["keyframes"]],
            [EventMarker.from_dict(e) for e in d["events"]],
            loop=d.get("loop", False), tail_blend=d.get("tail_blend", 0.0),
        )
