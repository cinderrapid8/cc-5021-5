import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stageactor.authoring import (
    SCRIPT_SECONDS, build_bundle, build_gesture_clip, build_idle_clip,
    build_script, build_walk_in_clip, build_exit_clip,
)
from stageactor.checks import (
    assert_no_pose_jumps, compressed_state_path,
    transition_boundary_continuity,
)
from stageactor.clip import AnimationClip, EventMarker, Keyframe
from stageactor.cli import run_demo, verify_roundtrip
from stageactor.graph import Bundle, Condition, State, StateGraph, Transition
from stageactor.machine import ActorMachine
from stageactor.replay import ReplayRunner
from stageactor.skeleton import fk

DT = 1.0 / 30.0
FRAMES = round(SCRIPT_SECONDS * 30.0)


def fresh_runner(seek_policy=ActorMachine.SEEK_SUPPRESS):
    bundle = build_bundle()
    machine = ActorMachine(bundle, seek_policy=seek_policy)
    runner = ReplayRunner(machine, dt=DT)
    runner.run(build_script(DT), FRAMES)
    return runner, machine


class ReplayDeterminismTests(unittest.TestCase):
    def test_state_path_covers_all_five_states(self):
        runner, _ = fresh_runner()
        path = compressed_state_path(runner.export_dict()["frames"])
        names = [p[0] for p in path]
        self.assertEqual(
            names,
            ["waiting", "walking_in", "presenting", "exiting", "offstage"],
        )

    def test_two_runs_produce_identical_frames(self):
        r1, _ = fresh_runner()
        r2, _ = fresh_runner()
        d1, d2 = r1.export_dict(), r2.export_dict()
        self.assertEqual(d1["frames"], d2["frames"])
        self.assertEqual(d1["all_events"], d2["all_events"])

    def test_no_pose_jump_across_any_transition(self):
        runner, _ = fresh_runner()
        frames = runner.export_dict()["frames"]
        # 核心判据：过渡边界位移相对前一帧不得突增（骨骼重置必然表现为突增；
        # 正常循环/走步运动在切换前后是同量级平滑曲线）
        boundaries = transition_boundary_continuity(frames)
        self.assertTrue(boundaries)
        for frame, kind, delta, prev in boundaries:
            self.assertLessEqual(
                delta, prev * 1.25 + 1e-6,
                f"过渡{kind}帧 {frame} 位移 {delta:.4f} 相对前帧 {prev:.4f} 突增")
        # 首个过渡从静止 idle 起步，进入帧必须接近零位移
        first_enter = next(d for (f, k, d, _p) in boundaries if k == "enter")
        self.assertLess(first_enter, 0.02)
        # 全程最大逐帧位移应是平滑物理运动（走步），而非瞬时大跳
        worst, where = assert_no_pose_jumps(frames)
        self.assertLess(worst, 0.6)

    def test_saved_graph_roundtrip_is_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner, report, graph_path, _ = run_demo(tmp)
            runner2, report2 = verify_roundtrip(graph_path)
            self.assertEqual(
                runner.export_dict()["frames"],
                runner2.export_dict()["frames"],
            )
            self.assertEqual(report["state_path"], report2["state_path"])
            self.assertEqual(report["event_total"], report2["event_total"])


class ClipSemanticsTests(unittest.TestCase):
    def test_loop_tail_blend_matches_cyclic_neighbor(self):
        clip = build_idle_clip()
        d = clip.tail_blend
        # 尾部窗口起点与紧邻前一时刻姿态应连续（根位置不重置）
        p_before, _, _ = clip.sample(clip.duration - d - 1e-6)
        p_enter, _, _ = clip.sample(clip.duration - d)
        self.assertAlmostEqual(
            math.dist(p_before.root_pos, p_enter.root_pos), 0.0, places=9)
        # 相位回绕：duration 处采样相位归零，且不是把骨骼重置成首帧 T-pose
        p_wrap, phase, finished = clip.sample(clip.duration)
        self.assertAlmostEqual(phase, 0.0)
        self.assertFalse(finished)
        p_zero, _, _ = clip.sample(0.0)
        self.assertAlmostEqual(
            math.dist(p_wrap.root_pos, p_zero.root_pos), 0.0, places=9)
        # 尾部窗口内输出是两套采样的混合，权重边界与窗口外一致
        p_mid, _, _ = clip.sample(clip.duration - d * 0.5)
        self.assertEqual(len(p_mid.rotations), len(p_zero.rotations))

    def test_non_loop_holds_last_pose_after_end(self):
        clip = build_walk_in_clip()
        p_end, _, finished = clip.sample(clip.duration)
        self.assertTrue(finished)
        p_later, phase, finished2 = clip.sample(clip.duration + 5.0)
        self.assertTrue(finished2)
        self.assertEqual(phase, clip.duration)
        self.assertEqual(p_end.root_pos, p_later.root_pos)
        self.assertEqual(set(p_end.rotations), set(p_later.rotations))

    def test_negative_time_wraps_into_previous_cycle(self):
        clip = build_gesture_clip()
        # 取在尾部混合窗口之外的时刻，验证负时间回绕到上一周期同一相位
        probe = 0.5
        p_neg, phase, _ = clip.sample(-probe)
        self.assertAlmostEqual(phase, clip.duration - probe)
        p_ref, _, _ = clip.sample(clip.duration - probe)
        self.assertAlmostEqual(
            math.dist(p_neg.root_pos, p_ref.root_pos), 0.0, places=9)


class EventPolicyTests(unittest.TestCase):
    def test_footstep_count_matches_markers(self):
        runner, machine = fresh_runner()
        # walk_in 4 个脚步 + exit 4 个脚步
        self.assertEqual(runner.event_count("footstep"), 8)

    def test_one_frame_skipping_multiple_markers_fires_all(self):
        bundle = build_bundle()
        machine = ActorMachine(bundle)
        machine.set_variable("entry_cue", True)
        machine.step(DT)                       # 进入 walking_in
        # 单步跨过 walk_in 的前三个脚步 (0.3/0.9/1.5)
        machine.step(1.6)
        names = [e.marker.name for e in machine.all_events]
        self.assertGreaterEqual(names.count("footstep"), 3)

    def test_pause_freezes_clocks_and_emits_nothing(self):
        bundle = build_bundle()
        machine = ActorMachine(bundle)
        before = len(machine.all_events)
        machine.pause()
        for _ in range(30):
            machine.step(DT)
        self.assertEqual(len(machine.all_events), before)
        self.assertEqual(machine.master_time, 0.0)
        machine.resume()
        machine.step(DT)
        self.assertAlmostEqual(machine.master_time, DT)

    def test_rate_change_does_not_drop_or_duplicate_events(self):
        bundle = build_bundle()
        machine = ActorMachine(bundle)
        machine.set_variable("entry_cue", True)
        machine.step(DT)
        machine.set_rate(2.0)
        for _ in range(20):                    # 实际推进 20*2*dt ≈ 1.333s
            machine.step(DT)
        walk = [e for e in machine.all_events
                if e.clip_name == "walk_in" and e.marker.name == "footstep"]
        # 覆盖到 0.3/0.9，第三个标记 1.5 尚未到达；每个事件 id 唯一
        self.assertEqual(len(walk), 2)
        ids = [e.event_id for e in walk]
        self.assertEqual(len(ids), len(set(ids)))

    def test_seek_suppress_does_not_replay_markers(self):
        runner, machine = fresh_runner()
        seek_events = [e for e in machine.all_events if e.reason == "seek_replay"]
        self.assertEqual(seek_events, [])

    def test_seek_replay_emits_crossed_markers_once(self):
        bundle = build_bundle()
        machine = ActorMachine(bundle, seek_policy=ActorMachine.SEEK_REPLAY)
        # 直接进入 walking_in 并把局部时钟放到 0.05，隔离 seek 语义
        machine.set_variable("entry_cue", True)
        machine.step(DT)
        machine.transition = None
        machine.current_state = "walking_in"
        machine.active = [machine.active[1]]
        machine.active[0].weight = 1.0
        machine.active[0].local_time = 0.05

        machine.seek(0.8)                     # 区间 (0.05, 0.8] 仅含脚步 0.3
        replayed = [e for e in machine.all_events
                    if e.reason == "seek_replay"]
        self.assertEqual(len(replayed), 1)
        self.assertEqual(replayed[0].marker.data["foot"], "left")
        # 已补发标记被登记，后续正常推进不能再次触发
        before = len(machine.all_events)
        machine.step(0.05)
        new_steps = [e for e in machine.all_events[before:]
                     if e.marker.name == "footstep"]
        self.assertEqual(new_steps, [])
        # 继续推进到 0.9 之后，第二个脚步正常触发且 id 不同
        machine.step(0.2)
        later = [e for e in machine.all_events
                 if e.marker.name == "footstep"]
        self.assertEqual(len(later), 2)
        self.assertEqual(len({e.event_id for e in later}), 2)

    def test_backward_seek_rearms_earlier_markers(self):
        bundle = build_bundle()
        machine = ActorMachine(bundle, seek_policy=ActorMachine.SEEK_REPLAY)
        machine.set_variable("entry_cue", True)
        machine.step(DT)
        machine.seek(1.0)
        self.assertGreaterEqual(
            len([e for e in machine.all_events if e.reason == "seek_replay"]), 2)
        machine.seek(0.05)                    # 后退：已触发集合重写
        machine.seek(1.0)                     # 再次前进应重新补发
        replayed = [e for e in machine.all_events
                    if e.reason == "seek_replay"]
        self.assertGreaterEqual(len(replayed), 4)

    def test_end_to_end_seek_policies_deterministically_differ(self):
        from stageactor.replay import ReplayRunner
        sup = ReplayRunner(ActorMachine(build_bundle(),
                                        seek_policy=ActorMachine.SEEK_SUPPRESS), dt=DT)
        rep = ReplayRunner(ActorMachine(build_bundle(),
                                        seek_policy=ActorMachine.SEEK_REPLAY), dt=DT)
        script = build_script(DT)
        sup.run(script, FRAMES)
        rep.run(script, FRAMES)
        sup_replay = [e for e in sup.machine.all_events
                      if e.reason == "seek_replay"]
        rep_replay = [e for e in rep.machine.all_events
                      if e.reason == "seek_replay"]
        self.assertEqual(sup_replay, [])
        self.assertGreater(len(rep_replay), 0)
        # 除补发事件外，姿态样本与状态路径完全一致
        self.assertEqual(sup.state_path(), rep.state_path())
        self.assertEqual(
            [f["joint_positions"] for f in sup.export_dict()["frames"]],
            [f["joint_positions"] for f in rep.export_dict()["frames"]],
        )


class TransitionMechanicsTests(unittest.TestCase):
    def test_weights_cross_fade_during_transition(self):
        runner, _ = fresh_runner()
        frames = runner.export_dict()["frames"]
        active = [f for f in frames if f["transition_target"] is not None]
        self.assertTrue(active)
        first = active[0]
        self.assertEqual(len(first["weights"]), 2)
        # 过渡首帧源权重接近 1、目标接近 0，且权重和为 1
        w = list(first["weights"].values())
        self.assertAlmostEqual(sum(w), 1.0, places=6)
        self.assertGreater(w[0], 0.9)

    def test_phase_match_continues_local_time(self):
        bundle = build_bundle()
        machine = ActorMachine(bundle)
        machine.set_variable("entry_cue", True)
        # 推进到 walk_in 自然结束并进入 presenting
        for _ in range(FRAMES):
            machine.step(DT)
            if machine.current_state == "presenting" and machine.transition is None:
                break
        # presenting 是循环片段，局部时间不应从 0 重新起步
        self.assertGreater(machine.active[0].local_time, 0.0)

    def test_resume_phase_mode_restores_saved_local_time(self):
        # 构造 a->b(restart)->a(resume)：回到 a 时应恢复离开 a 时记忆的局部时间
        from stageactor.authoring import (default_skeleton, build_idle_clip,
                                          build_gesture_clip)
        from stageactor.graph import Bundle
        skel = default_skeleton()
        idle, gesture = build_idle_clip(), build_gesture_clip()
        states = [State("a", "idle"), State("b", "gesture")]
        transitions = [
            Transition("a", "b", [Condition("go", "==", True)], 0.1,
                       phase_mode="restart"),
            Transition("b", "a", [Condition("back", "==", True)], 0.1,
                       phase_mode="resume"),
        ]
        graph = StateGraph(states, transitions, "a")
        machine = ActorMachine(Bundle(skel, [idle, gesture], graph))
        # 先在 a 停留，使 a 记住非零局部时间
        for _ in range(12):
            machine.step(DT)
        self.assertGreater(machine.resume_map["a"], 0.0)
        remembered = machine.resume_map["a"]
        machine.set_variable("go", True)
        machine.step(DT)
        machine.transition = None
        machine.current_state = "b"
        machine.active = [machine.active[1]]
        machine.active[0].weight = 1.0
        # 在 b 停留一段时间，再回 a（resume 应恢复 a 离开时记忆的局部时间）
        for _ in range(10):
            machine.step(DT)
        machine.set_variable("back", True)
        machine.step(DT)
        target = machine.active[1]
        self.assertEqual(target.clip_name, "idle")
        # resume 恢复 a 的记忆局部时间（循环片段按相位等价比较，容一个帧步长）
        self.assertAlmostEqual(
            target.local_time % idle.duration,
            remembered % idle.duration, delta=DT * 1.01)
        # 关键：不是 restart 的 0，也不是 b 的 gesture 时钟
        self.assertGreater(target.local_time, 0.1)

    def test_transition_cue_event_fires_once(self):
        runner, _ = fresh_runner()
        cues = [e for e in runner.export_dict()["all_events"]
                if e["name"] == "entry_signal"]
        self.assertEqual(len(cues), 1)

    def test_root_offsets_keep_world_position_continuous(self):
        # walk_in 在台心结束，过渡到 presenting 后世界根 x 应保持在 0 附近
        runner, _ = fresh_runner()
        frames = runner.export_dict()["frames"]
        for f in frames:
            if f["state"] == "presenting":
                root = f["joint_positions"][0]
                self.assertAlmostEqual(root[0], 0.0, places=4)
                break

    def test_no_skeleton_reset_at_switch(self):
        # 每个过渡进入帧的位移都不得相对其前一帧突增（目标权重为 0，无重置）
        runner, _ = fresh_runner()
        frames = runner.export_dict()["frames"]
        enters = [(f, d, p) for (f, k, d, p)
                  in transition_boundary_continuity(frames) if k == "enter"]
        self.assertTrue(enters)
        for frame, d, prev in enters:
            self.assertLessEqual(d, prev * 1.25 + 1e-6,
                                 f"进入帧 {frame} 出现重置突增")


class SkeletonEvaluationTests(unittest.TestCase):
    def test_fk_moves_child_joints_with_bone_rotation(self):
        bundle = build_bundle()
        skel = bundle.skeleton
        from stageactor.math3d import qfrom_axis_angle
        from stageactor.skeleton import Pose, identity_pose
        rest = identity_pose(skel)
        p0, _ = fk(skel, rest)
        moved = Pose((0, 0, 0), dict(rest.rotations))
        moved.rotations["r_shoulder"] = qfrom_axis_angle((0, 0, 1), 1.2)
        p1, _ = fk(skel, moved)
        # 肘部与手部世界位置必须随肩部旋转改变
        i_elbow = skel.index["r_elbow"]
        i_hand = skel.index["r_hand"]
        self.assertGreater(math.dist(p0[i_elbow], p1[i_elbow]), 0.05)
        self.assertGreater(math.dist(p0[i_hand], p1[i_hand]), 0.1)
        # 根骨骼位置不变
        self.assertEqual(p0[0], p1[0])

    def test_blend_weights_sum_semantics(self):
        runner, _ = fresh_runner()
        for f in runner.export_dict()["frames"]:
            self.assertAlmostEqual(sum(f["weights"].values()), 1.0, places=5)


class ViewerExportTests(unittest.TestCase):
    def test_viewer_html_embeds_replay_and_is_self_contained(self):
        import json as _json
        import re
        from stageactor.viewer import build_viewer
        runner, _ = fresh_runner()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "viewer.html")
            build_viewer(runner.export_dict(), path)
            with open(path, encoding="utf-8") as fh:
                html = fh.read()
            self.assertNotIn("window.REPLAY = ;", html)
            m = re.search(r"window\.REPLAY = (\{.*?\});", html, re.S)
            self.assertIsNotNone(m)
            embedded = _json.loads(m.group(1))
            self.assertEqual(len(embedded["frames"]),
                             len(runner.export_dict()["frames"]))
            # 单文件自包含：不引用外部脚本
            self.assertNotIn('<script src=', html)


if __name__ == "__main__":
    unittest.main(verbosity=2)
