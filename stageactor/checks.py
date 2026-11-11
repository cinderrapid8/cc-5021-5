"""回放结果的可复用核对工具（测试与 CLI 共用）。"""

import math


def joint_distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def max_pose_delta(rec_a, rec_b):
    return max(joint_distance(a, b)
               for a, b in zip(rec_a["joint_positions"],
                               rec_b["joint_positions"]))


def compressed_state_path(frames):
    """把逐帧状态压缩成 (状态, 起始帧) 路径。"""
    path = []
    for fr in frames:
        if not path or path[-1][0] != fr["state"]:
            path.append((fr["state"], fr["frame"]))
    return path


def assert_no_pose_jumps(frames, threshold=0.15):
    """逐帧世界关节位移不得超过阈值（骨骼不重置、无跳变）。

    seek 帧是显式时间跳转，允许不连续，因此 seek 当帧与下一帧的比较被豁免；
    正常推进（含过渡混合）路径上任何跳变都会被报出。
    """
    worst = 0.0
    where = 0
    for i in range(1, len(frames)):
        if frames[i].get("seeked") or frames[i - 1].get("seeked"):
            continue
        a, b = frames[i - 1], frames[i]
        d = max_pose_delta(a, b)
        if d > worst:
            worst, where = d, b["frame"]
    return worst, where


def transition_boundary_continuity(frames):
    """过渡进入/离开边界的连续性检查数据。

    返回 (帧, enter/leave, 边界位移, 前一帧位移)。真正的骨骼重置表现为
    边界位移相对前一帧突增；正常走步的高速根运动在过渡前后是同量级平滑曲线。
    """
    results = []
    for i in range(1, len(frames)):
        a, b = frames[i - 1], frames[i]
        entering = (a["transition_target"] is None
                    and b["transition_target"] is not None)
        leaving = (a["transition_target"] is not None
                   and b["transition_target"] is None)
        if (entering or leaving) and not b.get("seeked"):
            prev = max_pose_delta(frames[i - 2], a) if i >= 2 else 0.0
            results.append((b["frame"],
                            "enter" if entering else "leave",
                            max_pose_delta(a, b), prev))
    return results


def events_named(frames, name):
    out = []
    for fr in frames:
        for ev in fr["events"]:
            if ev["name"] == name:
                out.append((fr["frame"], ev))
    return out


def total_event_count(export, name=None):
    evs = export["all_events"]
    if name is None:
        return len(evs)
    return sum(1 for e in evs if e["name"] == name)


def weights_during(frames, state=None):
    out = []
    for fr in frames:
        if state is None or fr["state"] == state:
            out.append((fr["frame"], fr["weights"]))
    return out
