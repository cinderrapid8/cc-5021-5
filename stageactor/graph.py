"""状态图：状态、条件、过渡（持续时间与相位同步规则分别建模）。

条件是确定性谓词，只依赖输入变量字典，不依赖实时随机源，因此保存状态图后
重新运行可得到完全一致的状态路径。
"""


class Condition:
    """形如 var op value 的原子条件；AND 用列表组合（all 语义）。"""

    OPS = ("==", "!=", ">", ">=", "<", "<=")

    def __init__(self, variable, op, value):
        if op not in self.OPS:
            raise ValueError(f"未知比较运算符: {op}")
        self.variable = variable
        self.op = op
        self.value = value

    def evaluate(self, variables):
        v = variables.get(self.variable)
        op, w = self.op, self.value
        if op == "==":
            return v == w
        if op == "!=":
            return v != w
        if v is None:
            return False
        if op == ">":
            return v > w
        if op == ">=":
            return v >= w
        if op == "<":
            return v < w
        return v <= w

    def describe(self):
        return f"{self.variable} {self.op} {self.value!r}"

    def to_dict(self):
        return {"variable": self.variable, "op": self.op, "value": self.value}

    @classmethod
    def from_dict(cls, d):
        return cls(d["variable"], d["op"], d["value"])


class Transition:
    """状态过渡；与动画采样时间分开建模。

    phase_mode:
      * restart      目标片段从局部时间 0 开始；
      * phase_match  目标局部时间 = 源片段当前相位（用于等长循环衔接）；
      * resume       目标片段从每个状态各自记忆的局部时间恢复。
    cue: 可选，过渡在该条件成立的瞬间抛出的提示事件名。
    """

    PHASE_MODES = ("restart", "phase_match", "resume")

    def __init__(self, source, target, conditions, duration,
                 phase_mode="restart", cue=None, priority=0):
        if phase_mode not in self.PHASE_MODES:
            raise ValueError(f"未知相位同步规则: {phase_mode}")
        self.source = source
        self.target = target
        self.conditions = list(conditions)
        self.duration = float(duration)
        self.phase_mode = phase_mode
        self.cue = cue
        self.priority = int(priority)

    def evaluate(self, variables):
        return all(c.evaluate(variables) for c in self.conditions)

    def basis(self):
        return " AND ".join(c.describe() for c in self.conditions)

    def to_dict(self):
        return {
            "source": self.source, "target": self.target,
            "conditions": [c.to_dict() for c in self.conditions],
            "duration": self.duration, "phase_mode": self.phase_mode,
            "cue": self.cue, "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            d["source"], d["target"],
            [Condition.from_dict(c) for c in d["conditions"]],
            d["duration"], phase_mode=d.get("phase_mode", "restart"),
            cue=d.get("cue"), priority=d.get("priority", 0),
        )


class State:
    def __init__(self, name, clip_name, auto_finish=False):
        self.name = name
        self.clip_name = clip_name
        self.auto_finish = bool(auto_finish)

    def to_dict(self):
        return {"name": self.name, "clip_name": self.clip_name,
                "auto_finish": self.auto_finish}

    @classmethod
    def from_dict(cls, d):
        return cls(d["name"], d["clip_name"], d.get("auto_finish", False))


class StateGraph:
    def __init__(self, states, transitions, initial_state):
        self.states = {s.name: s for s in states}
        if initial_state not in self.states:
            raise ValueError("初始状态不在状态表中")
        self.transitions = list(transitions)
        self.initial_state = initial_state
        self._out = {}
        for tr in self.transitions:
            self._out.setdefault(tr.source, []).append(tr)
        for outs in self._out.values():
            outs.sort(key=lambda t: -t.priority)

    def outgoing(self, state_name):
        return self._out.get(state_name, [])

    def to_dict(self):
        return {
            "states": [s.to_dict() for s in self.states.values()],
            "transitions": [t.to_dict() for t in self.transitions],
            "initial_state": self.initial_state,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            [State.from_dict(s) for s in d["states"]],
            [Transition.from_dict(t) for t in d["transitions"]],
            d["initial_state"],
        )


class Bundle:
    """状态图 + 片段 + 骨骼的可保存整包；重新加载后回放结果一致。"""

    VERSION = 1

    def __init__(self, skeleton, clips, graph):
        self.skeleton = skeleton
        self.clips = {c.name: c for c in clips}
        self.graph = graph

    def to_dict(self):
        return {
            "version": self.VERSION,
            "skeleton": self.skeleton.to_dict(),
            "clips": [c.to_dict() for c in self.clips.values()],
            "graph": self.graph.to_dict(),
        }

    @classmethod
    def from_dict(cls, d):
        from .skeleton import Skeleton
        from .clip import AnimationClip
        skel = Skeleton.from_dict(d["skeleton"])
        clips = [AnimationClip.from_dict(c) for c in d["clips"]]
        graph = StateGraph.from_dict(d["graph"])
        return cls(skel, clips, graph)
