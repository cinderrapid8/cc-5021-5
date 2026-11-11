"""自制内容：默认人形骨骼、四段程序化动画、舞台状态图与固定回放脚本。

所有动画由关键帧姿态在代码中构造（不是预录视频），姿态经 FK 驱动真实骨骼。
"""

import math

from .clip import AnimationClip, EventMarker, Keyframe
from .graph import Condition, State, StateGraph, Transition
from .math3d import qfrom_axis_angle, qidentity, qmul
from .skeleton import Bone, Pose, Skeleton


# ============================================================ 默认人形骨骼
def default_skeleton():
    # 坐标系：x 指向演员左侧，y 向上，z 面向观众（前方）
    bones = [
        Bone("root", -1, (0, 1, 0), 0.0),
        Bone("spine", 0, (0, 1, 0), 0.55),
        Bone("chest", 1, (0, 1, 0), 0.35),
        Bone("neck", 2, (0, 1, 0), 0.18),
        Bone("head", 3, (0, 1, 0), 0.22),
        Bone("l_shoulder", 2, (1, 0, 0), 0.18),
        Bone("l_elbow", 5, (1, -0.1, 0), 0.30),
        Bone("l_hand", 6, (1, -0.05, 0), 0.26),
        Bone("r_shoulder", 2, (-1, 0, 0), 0.18),
        Bone("r_elbow", 8, (-1, -0.1, 0), 0.30),
        Bone("r_hand", 9, (-1, -0.05, 0), 0.26),
        Bone("l_hip", 0, (0.18, -1, 0), 0.10),
        Bone("l_knee", 11, (0, -1, 0.05), 0.45),
        Bone("l_foot", 12, (0, -0.1, 1), 0.45),
        Bone("r_hip", 0, (-0.18, -1, 0), 0.10),
        Bone("r_knee", 14, (0, -1, 0.05), 0.45),
        Bone("r_foot", 15, (0, -0.1, 1), 0.45),
    ]
    return Skeleton(bones)


# ============================================================ 姿态构造工具
def _pose(root_x, rots):
    valid = set(default_skeleton().names)
    full = {name: qidentity() for name in valid}
    for name, q in rots.items():
        if name in valid:
            full[name] = q
    return Pose((root_x, 0.0, 0.0), full)


def _ax(axis, angle):
    return qfrom_axis_angle(axis, angle)


def _chain(*rots):
    q = qidentity()
    for r in rots:
        q = qmul(q, r)
    return q


# ============================================================ 1. 待场（循环）
def build_idle_clip():
    duration = 1.6
    keys = []
    steps = 8
    for i in range(steps + 1):
        t = duration * i / steps
        s = math.sin(2 * math.pi * i / steps)          # 呼吸正弦
        c = math.cos(2 * math.pi * i / steps)
        rots = {
            "chest": _ax((1, 0, 0), 0.03 * s),
            "head": _ax((1, 0, 0), -0.04 * s),
            "l_shoulder": _ax((0, 0, 1), -0.05 * s),
            "r_shoulder": _ax((0, 0, 1), 0.05 * s),
            "l_knee": _ax((1, 0, 0), 0.02 * c),
            "r_knee": _ax((1, 0, 0), -0.02 * c),
        }
        keys.append(Keyframe(t, _pose(0.0, rots)))
    events = [
        EventMarker(0.0, "breath_cue", {"phase": "inhale"}),
        EventMarker(0.8, "breath_cue", {"phase": "exhale"}),
    ]
    return AnimationClip("idle", duration, keys, events,
                         loop=True, tail_blend=0.2)


# ============================================================ 2. 走入（非循环）
def build_walk_in_clip():
    duration = 2.4
    n = 8
    keys = []
    for i in range(n + 1):
        t = duration * i / n
        phase = 2 * math.pi * i / n
        root_x = -3.2 * (1.0 - i / n)                  # 从侧台 (-3.2) 走到台心 (0)
        swing = math.sin(phase)
        rots = {
            "l_hip": _ax((1, 0, 0), 0.5 * swing),
            "r_hip": _ax((1, 0, 0), -0.5 * swing),
            "l_knee": _ax((1, 0, 0), max(0.0, 0.6 * -swing)),
            "r_knee": _ax((1, 0, 0), max(0.0, 0.6 * swing)),
            "l_shoulder": _ax((1, 0, 0), -0.4 * swing),
            "r_shoulder": _ax((1, 0, 0), 0.4 * swing),
            "chest": _ax((0, 1, 0), 0.06 * math.cos(phase)),
        }
        keys.append(Keyframe(t, _pose(root_x, rots)))
    events = [
        EventMarker(0.3, "footstep", {"foot": "left"}),
        EventMarker(0.9, "footstep", {"foot": "right"}),
        EventMarker(1.5, "footstep", {"foot": "left"}),
        EventMarker(2.1, "footstep", {"foot": "right"}),
        EventMarker(2.35, "arrival_cue", {"at": "center_mark"}),
    ]
    return AnimationClip("walk_in", duration, keys, events, loop=False)


# ============================================================ 3. 站定示意（循环）
def build_gesture_clip():
    duration = 2.0
    keys = []
    n = 10
    for i in range(n + 1):
        t = duration * i / n
        s = math.sin(2 * math.pi * i / n)
        raise_r = max(0.0, s)                            # 右手示意举起
        raise_l = max(0.0, -s)
        rots = {
            "r_shoulder": _chain(_ax((0, 0, 1), -1.2 * raise_r),
                                 _ax((1, 0, 0), 0.3 * raise_r)),
            "r_elbow": _ax((0, 0, 1), -0.5 * raise_r),
            "l_shoulder": _chain(_ax((0, 0, 1), 1.2 * raise_l),
                                 _ax((1, 0, 0), 0.3 * raise_l)),
            "l_elbow": _ax((0, 0, 1), 0.5 * raise_l),
            "head": _ax((0, 1, 0), 0.25 * s),
            "chest": _ax((1, 0, 0), 0.03 * s),
        }
        keys.append(Keyframe(t, _pose(0.0, rots)))
    events = [
        EventMarker(0.5, "gesture_cue", {"hand": "right"}),
        EventMarker(1.5, "gesture_cue", {"hand": "left"}),
    ]
    return AnimationClip("gesture", duration, keys, events,
                         loop=True, tail_blend=0.2)


# ============================================================ 4. 离场（非循环）
def build_exit_clip():
    duration = 2.6
    n = 8
    keys = []
    for i in range(n + 1):
        t = duration * i / n
        phase = 2 * math.pi * i / n
        root_x = 3.2 * i / n                             # 从台心走到对侧台 (+3.2)
        swing = math.sin(phase)
        bow = max(0.0, 1.0 - i / 2.0) if i <= 2 else 0.0
        rots = {
            "l_hip": _ax((1, 0, 0), -0.5 * swing),
            "r_hip": _ax((1, 0, 0), 0.5 * swing),
            "l_knee": _ax((1, 0, 0), max(0.0, 0.6 * swing)),
            "r_knee": _ax((1, 0, 0), max(0.0, 0.6 * -swing)),
            "l_shoulder": _ax((1, 0, 0), 0.4 * swing),
            "r_shoulder": _ax((1, 0, 0), -0.4 * swing),
            "chest": _ax((1, 0, 0), 0.5 * bow),
            "head": _ax((1, 0, 0), 0.4 * bow),
        }
        keys.append(Keyframe(t, _pose(root_x, rots)))
    events = [
        EventMarker(0.15, "bow_cue", {"depth": "half"}),
        EventMarker(0.6, "footstep", {"foot": "left"}),
        EventMarker(1.2, "footstep", {"foot": "right"}),
        EventMarker(1.8, "footstep", {"foot": "left"}),
        EventMarker(2.4, "footstep", {"foot": "right"}),
    ]
    return AnimationClip("exit", duration, keys, events, loop=False)


def build_gone_clip():
    # 离场终点保持：极短非循环片段，结束后继续保持末姿态
    pose = _pose(3.2, {})
    return AnimationClip("gone", 0.2,
                         [Keyframe(0.0, pose), Keyframe(0.2, pose)],
                         [EventMarker(0.0, "exit_complete", {})],
                         loop=False)


# ============================================================ 状态图
def build_graph():
    states = [
        State("waiting", "idle"),
        State("walking_in", "walk_in", auto_finish=True),
        State("presenting", "gesture"),
        State("exiting", "exit", auto_finish=True),
        State("offstage", "gone", auto_finish=True),
    ]
    transitions = [
        Transition("waiting", "walking_in",
                   [Condition("entry_cue", "==", True)],
                   duration=0.35, phase_mode="restart",
                   cue="entry_signal"),
        Transition("walking_in", "presenting",
                   [Condition("__finished__", "==", True)],
                   duration=0.4, phase_mode="phase_match",
                   cue="arrival_signal"),
        Transition("presenting", "exiting",
                   [Condition("exit_cue", "==", True)],
                   duration=0.35, phase_mode="restart",
                   cue="exit_signal"),
        Transition("exiting", "offstage",
                   [Condition("__finished__", "==", True)],
                   duration=0.25, phase_mode="restart"),
    ]
    return StateGraph(states, transitions, "waiting")


def build_bundle():
    from .graph import Bundle
    skel = default_skeleton()
    clips = [build_idle_clip(), build_walk_in_clip(), build_gesture_clip(),
             build_exit_clip(), build_gone_clip()]
    return Bundle(skel, clips, build_graph())


# ============================================================ 固定回放脚本
def build_script(dt=1.0 / 30.0):
    """30Hz 固定脚本：覆盖进入、暂停/恢复、变速、示意、跳转、离场。"""
    f = lambda sec: round(sec / dt)
    return [
        {"at": f(1.0), "cmd": "set", "name": "entry_cue", "value": True},
        {"at": f(3.2), "cmd": "pause"},
        {"at": f(4.2), "cmd": "resume"},
        {"at": f(5.0), "cmd": "rate", "value": 1.5},
        {"at": f(7.0), "cmd": "rate", "value": 1.0},
        # 前进跳转：replay 政策补发跨过的 gesture_cue，suppress 政策不补发
        {"at": f(7.6), "cmd": "seek", "time": 8.7},
        # 后退跳转：两种政策都不补发，仅重写已触发集合
        {"at": f(8.5), "cmd": "seek", "time": 6.0},
        {"at": f(10.5), "cmd": "set", "name": "exit_cue", "value": True},
        {"at": f(15.0), "cmd": "set", "name": "entry_cue", "value": False},
    ]


SCRIPT_SECONDS = 17.0
