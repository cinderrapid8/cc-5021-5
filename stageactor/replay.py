"""固定输入序列回放：不依赖人工连续点击，同一份脚本永远产生同一条轨迹。

输入脚本是有序步骤列表，每步 {"at": 帧序号, "cmd": ..., ...}，按帧号触发；
回放器以固定 dt 推进 ActorMachine，逐帧记录状态、混合权重、触发依据、
事件与世界空间关节位置，并导出 JSON 供界面与断言核对。
"""

import json


class FrameRecord:
    def __init__(self, frame, master_time, state, transition_target,
                 transition_progress, weights, basis, events, joint_positions,
                 variables, paused, rate, seeked):
        self.frame = frame
        self.master_time = master_time
        self.state = state
        self.transition_target = transition_target
        self.transition_progress = transition_progress
        self.weights = weights
        self.basis = basis
        self.events = events
        self.joint_positions = joint_positions
        self.variables = variables
        self.paused = paused
        self.rate = rate
        self.seeked = seeked

    def to_dict(self):
        return {
            "frame": self.frame, "time": round(self.master_time, 6),
            "state": self.state,
            "transition_target": self.transition_target,
            "transition_progress": self.transition_progress,
            "weights": self.weights, "basis": self.basis,
            "events": [e.to_dict() for e in self.events],
            "joint_positions": [[round(v, 6) for v in p]
                                for p in self.joint_positions],
            "variables": _jsonable(self.variables),
            "paused": self.paused, "rate": self.rate, "seeked": self.seeked,
        }


def _jsonable(values):
    out = {}
    for k, v in values.items():
        out[k] = v if isinstance(v, (bool, int, float, str)) or v is None else str(v)
    return out


class ReplayRunner:
    def __init__(self, machine, dt=1.0 / 30.0):
        self.machine = machine
        self.dt = dt
        self.frames = []
        self.script = []
        self._seek_frames = set()

    def run(self, script, frame_count):
        """执行固定脚本。支持的命令：
        set / pause / resume / rate / seek。
        """
        self.script = list(script)
        by_frame = {}
        for step in script:
            by_frame.setdefault(step["at"], []).append(step)

        for f in range(frame_count):
            seeked = False
            for step in by_frame.get(f, []):
                if step["cmd"] == "seek":
                    seeked = True
                self._apply(step)
            if seeked:
                self._seek_frames.add(f)
            frame_events = self.machine.step(self.dt)
            self._record(f, frame_events, seeked)
        return self.frames

    def _apply(self, step):
        m = self.machine
        cmd = step["cmd"]
        if cmd == "set":
            m.set_variable(step["name"], step["value"])
        elif cmd == "pause":
            m.pause()
        elif cmd == "resume":
            m.resume()
        elif cmd == "rate":
            m.set_rate(step["value"])
        elif cmd == "seek":
            m.seek(step["time"])
        else:
            raise ValueError(f"未知回放命令: {cmd}")

    def _record(self, frame, events, seeked=False):
        m = self.machine
        positions, _, debug = m.evaluate()
        tr = m.transition
        self.frames.append(FrameRecord(
            frame=frame,
            master_time=m.master_time,
            state=m.current_state,
            transition_target=None if tr is None else tr[0].target,
            transition_progress=None if tr is None
            else round(tr[1] / tr[0].duration, 6),
            weights=m.weights(),
            basis=m.last_basis,
            events=events,
            joint_positions=positions,
            variables=dict(m.variables),
            paused=m.paused,
            rate=m.rate,
            seeked=seeked,
        ))

    # --------------------------------------------------------------- 导出
    def export_dict(self):
        skel = self.machine.skeleton
        return {
            "dt": self.dt,
            "joint_names": skel.names,
            "skeleton": skel.to_dict(),
            "script": self.script,
            "frames": [r.to_dict() for r in self.frames],
            "all_events": [e.to_dict() for e in self.machine.all_events],
            "seek_policy": self.machine.seek_policy,
        }

    def export_json(self, path):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.export_dict(), fh, ensure_ascii=False, indent=1)

    def state_path(self):
        return [r.state for r in self.frames]

    def event_count(self, name=None):
        evs = self.machine.all_events
        if name is None:
            return len(evs)
        return sum(1 for e in evs if e.marker.name == name)
