"""演员状态机：固定步长推进、过渡混合、事件触发策略。

三套时间分别建模：
  master_time   全局主时钟（秒），seek/pause/rate 只作用于它；
  过渡时钟      每条 Transition 自己的 duration/进度，与片段采样时间无关；
  局部采样时间  每个 ActiveClip 独立的 local_time，按 rate 推进并各自循环/保持。

切换语义：过渡启动时不重置任何骨骼，源片段保持求值并线性淡出，目标片段按
相位规则初始化后淡入；根位置差值被烘焙进目标片段时钟，保证空间连续。
"""

from .math3d import vscale, vsub
from .skeleton import blend_pose, fk, identity_pose


class ActiveClip:
    def __init__(self, clip_name, local_time, weight, root_offset):
        self.clip_name = clip_name
        self.local_time = local_time
        self.weight = weight
        self.root_offset = tuple(root_offset)   # 烘焙到根位置的空间偏移
        self.fired = set()                      # 已触发的 (周期, 标记索引)

    def state_key(self):
        return (self.clip_name, self.local_time, self.weight,
                self.root_offset, tuple(sorted(self.fired)))


class FiredEvent:
    def __init__(self, master_time, clip_name, marker, cycle, event_id, reason):
        self.master_time = master_time
        self.clip_name = clip_name
        self.marker = marker
        self.cycle = cycle
        self.event_id = event_id
        self.reason = reason                    # crossed / seek_replay / seek_suppressed

    def to_dict(self):
        return {
            "time": round(self.master_time, 6), "clip": self.clip_name,
            "name": self.marker.name, "cycle": self.cycle,
            "id": self.event_id, "reason": self.reason,
            "data": self.marker.data,
        }


class ActorMachine:
    SEEK_SUPPRESS = "suppress"
    SEEK_REPLAY = "replay"

    def __init__(self, bundle, seek_policy=SEEK_SUPPRESS):
        self.bundle = bundle
        self.skeleton = bundle.skeleton
        self.clips = bundle.clips
        self.graph = bundle.graph
        self.seek_policy = seek_policy

        self.master_time = 0.0
        self.rate = 1.0
        self.paused = False
        self.variables = {}
        self.active = []
        self.transition = None          # (Transition, 已过时间, 源, 目标, 目标时钟)
        self.current_state = self.graph.initial_state
        self.resume_map = {}            # 状态名 -> 离开时的局部时间
        self.events = []                # 本步新触发事件（每步清空）
        self.all_events = []
        self.last_basis = "初始状态"
        self._start_clip(self.graph.initial_state, 0.0,
                         self.clips[self.graph.states[self.graph.initial_state]
                                    .clip_name])
        self._publish_finish()

    # ================================================================ 内部工具
    def _start_clip(self, state_name, local_time, clip,
                    root_offset=(0, 0, 0)):
        self.active = [ActiveClip(clip.name, local_time, 1.0, root_offset)]
        self.resume_map[state_name] = local_time

    def _clip_of_state(self, state_name):
        return self.clips[self.graph.states[state_name].clip_name]

    def _init_target_clock(self, tr, source_clock):
        clip = self._clip_of_state(tr.target)
        if tr.phase_mode == "restart":
            return 0.0
        if tr.phase_mode == "phase_match":
            return source_clock.local_time % clip.duration
        return self.resume_map.get(tr.target, 0.0)

    # ================================================================ 指令接口
    def set_variable(self, name, value):
        self.variables[name] = value

    def set_rate(self, rate):
        # 只改变后续推进速度，不补发任何事件
        self.rate = float(rate)

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def seek(self, new_time):
        """时间跳转。delta>0 前进、delta<0 后退；事件政策由 seek_policy 决定。"""
        delta = float(new_time) - self.master_time
        if delta == 0.0:
            return
        for ac in self.active:
            clip = self.clips[ac.clip_name]
            old_t, new_t = ac.local_time, ac.local_time + delta
            if self.seek_policy == self.SEEK_REPLAY and delta > 0:
                self._collect_crossed(ac, clip, old_t, new_t, "seek_replay")
            self._apply_seek_fire_state(ac, clip, new_t, delta)
            ac.local_time = new_t
        self.master_time = float(new_time)
        if self.transition is not None:
            tr, elapsed, src, tgt, target_clock = self.transition
            target_clock = max(0.0, target_clock + delta)
            elapsed = max(0.0, min(tr.duration, elapsed + delta))
            self.transition = (tr, elapsed, src, tgt, target_clock)

    def _apply_seek_fire_state(self, ac, clip, new_t, delta):
        """跳转后按目标时间重写已触发集合。

        suppress 与后退：直接标记目标时间之前的全部事件为已触发（不补发）；
        replay 前进：跨区间事件已在 seek() 中即时补发，这里同样重写集合，
        保证后续正常推进既不重复也不漏发。
        """
        if clip.events:
            ac.fired = self._all_fired_before(clip, new_t)

    @staticmethod
    def _all_fired_before(clip, t):
        fired = set()
        for idx, m in enumerate(clip.events):
            if clip.loop:
                dur = clip.duration
                cyc = int(t // dur)
                rem = t % dur
                hit = cyc if m.time <= rem else cyc - 1
                for c in range(hit + 1):
                    fired.add((c, idx))
            elif m.time <= t:
                fired.add((0, idx))
        return fired

    # ================================================================ 事件触发
    def _collect_crossed(self, ac, clip, old_t, new_t, reason):
        """收集 (old_t, new_t] 区间内跨过的全部标记——一帧跨多个也不漏。"""
        if not clip.events or new_t <= old_t:
            return
        for idx, marker in enumerate(clip.events):
            if clip.loop:
                dur = clip.duration
                c0 = int(old_t // dur)
                c1 = int(new_t // dur)
                for c in range(c0, c1 + 1):
                    key = (c, idx)
                    mt = c * dur + marker.time
                    if old_t < mt <= new_t and key not in ac.fired:
                        self._fire(ac, clip, marker, c, reason, key=key,
                                   marker_index=idx)
            else:
                key = (0, idx)
                if old_t < marker.time <= new_t and key not in ac.fired:
                    self._fire(ac, clip, marker, 0, reason, key=key,
                               marker_index=idx)

    def _fire(self, ac, clip, marker, cycle, reason, key, marker_index):
        label = marker.name if marker_index is None else f"{marker.name}@{marker_index}"
        eid = f"{clip.name}:{label}#{cycle}"
        ev = FiredEvent(self.master_time, clip.name, marker, cycle, eid, reason)
        ac.fired.add(key)
        self.events.append(ev)
        self.all_events.append(ev)

    # ================================================================ 固定步长
    def step(self, dt, input_vars=None):
        """推进一帧；dt 为标称帧时长，实际推进 dt*rate。返回本帧事件列表。"""
        self.events = []
        if input_vars:
            self.variables.update(input_vars)

        if not self.paused:
            delta = dt * self.rate
            if delta != 0.0:
                self.master_time += delta
                self._advance_clocks(delta)
                self._update_transition(delta)
                self._publish_finish()
                self._check_transitions()
        else:
            # 暂停：时钟与过渡全部冻结，事件不补发也不丢失
            self._publish_finish()
            self._check_transitions()

        return list(self.events)

    def _publish_finish(self):
        # 非循环片段采样到末帧即置位；条件 "__finished__ == True" 驱动自动过渡
        ac = self.active[0]
        clip = self.clips[ac.clip_name]
        _, _, finished = clip.sample(ac.local_time)
        state = self.graph.states[self.current_state]
        self.variables["__finished__"] = bool(finished and state.auto_finish)

    def _advance_clocks(self, delta):
        if self.transition is not None:
            return  # 过渡中的两个时钟由 _update_transition 统一处理
        for ac in self.active:
            clip = self.clips[ac.clip_name]
            old_t = ac.local_time
            if not clip.loop and old_t >= clip.duration:
                continue  # 非循环片段结束即冻结，保持末帧
            new_t = old_t + delta
            self._collect_crossed(ac, clip, old_t, new_t, "crossed")
            ac.local_time = new_t
            # 持续更新当前状态的记忆局部时间，供 resume 相位规则恢复
            self.resume_map[self.current_state] = new_t

    def _update_transition(self, delta):
        if self.transition is None:
            return
        tr, elapsed, src, tgt, target_clock = self.transition
        elapsed += delta
        src_clock, tgt_clock = self.active[0], self.active[1]
        # 源片段继续按自身时钟推进；非循环片段到末帧后冻结，采样保持末姿态
        src_clip = self.clips[src_clock.clip_name]
        if src_clip.loop or src_clock.local_time < src_clip.duration:
            old_src = src_clock.local_time
            new_src = old_src + delta
            if not src_clip.loop:
                new_src = min(new_src, src_clip.duration)
            self._collect_crossed(src_clock, src_clip, old_src,
                                  new_src, "crossed")
            src_clock.local_time = new_src
        old_target = target_clock
        target_clock += delta
        self._collect_crossed(tgt_clock, self.clips[tgt_clock.clip_name],
                              old_target, target_clock, "crossed")
        tgt_clock.local_time = target_clock

        if elapsed >= tr.duration:
            # 过渡结束：源片段停止求值，目标权重置 1（无骨骼重置跳变）
            tgt_clock.weight = 1.0
            self.active = [tgt_clock]
            self.current_state = tgt
            self.resume_map[tgt] = tgt_clock.local_time
            self.transition = None
            self.last_basis = f"过渡完成 -> {tgt}"
            self._publish_finish()
            self._check_transitions()
        else:
            u = elapsed / tr.duration
            src_clock.weight = 1.0 - u
            tgt_clock.weight = u
            self.transition = (tr, elapsed, src, tgt, target_clock)

    # ================================================================ 状态过渡
    def _check_transitions(self):
        if self.transition is not None:
            return  # 过渡期间不抢占，保证确定性的状态路径
        for tr in self.graph.outgoing(self.current_state):
            if tr.evaluate(self.variables):
                self._begin_transition(tr)
                return

    def _begin_transition(self, tr):
        src_clock = self.active[0]
        src_clip = self.clips[src_clock.clip_name]
        tgt_clip = self._clip_of_state(tr.target)
        target_local = self._init_target_clock(tr, src_clock)

        # 根位置空间连续性：让目标片段加偏移后的世界根位置等于源当前世界根位置
        # 即 offset = src_world - tgt_local；偏移随目标权重衰减，过渡末归零
        src_pose, _, _ = src_clip.sample(src_clock.local_time)
        tgt_pose0, _, _ = tgt_clip.sample(target_local)
        offset = vsub(src_pose.root_pos, tgt_pose0.root_pos)

        tgt_clock = ActiveClip(tgt_clip.name, target_local, 0.0, offset)
        src_clock.weight = 1.0
        self.active = [src_clock, tgt_clock]
        self.resume_map[self.current_state] = src_clock.local_time
        self.transition = (tr, 0.0, self.current_state, tr.target, target_local)
        self.last_basis = f"{tr.source}->{tr.target} 当 {tr.basis()}"
        if tr.cue:
            marker = _CueMarker(tr.cue)
            self._fire(src_clock, src_clip, marker,
                       int(src_clock.local_time // max(src_clip.duration, 1e-9)),
                       "crossed",
                       key=("cue", tr.source, tr.target,
                            round(src_clock.local_time, 9)),
                       marker_index=None)

    # ================================================================ 姿态求值
    def evaluate(self):
        """真实骨骼求值：逐片段采样 -> 权重混合 -> FK，返回世界关节位置。

        根偏移在过渡创建时按“目标世界根 = 源当前世界根”烘焙，并随目标权重
        线性衰减：首帧目标权重 0，贡献完整偏移，世界根与源完全连续；末帧
        权重 1，偏移归零，完全落到目标自身根运动。整个过渡无骨骼重置跳变。
        """
        poses = []
        for ac in self.active:
            clip = self.clips[ac.clip_name]
            pose, phase, finished = clip.sample(ac.local_time)
            off = vscale(ac.root_offset, 1.0 - ac.weight)
            pose.root_pos = (
                pose.root_pos[0] + off[0],
                pose.root_pos[1] + off[1],
                pose.root_pos[2] + off[2],
            )
            poses.append((pose, ac.weight, phase, finished, ac.clip_name))

        if not poses:
            pose = identity_pose(self.skeleton)
            positions, _ = fk(self.skeleton, pose)
            return positions, pose, []

        base = poses[0][0]
        for pose, w, *_ in poses[1:]:
            base = blend_pose(self.skeleton, base, pose, w)
        positions, _ = fk(self.skeleton, base)
        debug = [
            {"clip": p[4], "weight": round(ac.weight, 6),
             "phase": round(p[2], 6), "finished": p[3]}
            for p, ac in zip(poses, self.active)
        ]
        return positions, base, debug

    def weights(self):
        return {ac.clip_name: round(ac.weight, 6) for ac in self.active}

    def state_key(self):
        return (
            round(self.master_time, 9), self.current_state, self.paused,
            self.rate, tuple(sorted(self.variables.items(), key=lambda kv: kv[0])),
            tuple(ac.state_key() for ac in self.active),
            None if self.transition is None else (
                self.transition[0].target, round(self.transition[1], 9)),
        )


class _CueMarker:
    """过渡提示事件复用 EventMarker 的字段结构。"""

    def __init__(self, name):
        self.time = 0.0
        self.name = name
        self.data = {"kind": "transition_cue"}
