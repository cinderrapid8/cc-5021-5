import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from stageactor.authoring import (  # noqa: E402
    build_bundle, build_gesture_clip, build_idle_clip,
)
from stageactor.graph import Bundle, Condition, State, StateGraph, Transition  # noqa: E402
from stageactor.machine import ActorMachine  # noqa: E402
from stageactor.release import (  # noqa: E402
    ReleaseError, default_runtime_config, load_package, produce_release,
    run_selfcheck,
)
from stageactor.runtime import start_package  # noqa: E402

DT = 1.0 / 30.0


class ReleasePackageTests(unittest.TestCase):
    def test_package_selfcheck_runs_old_sample_to_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = os.path.join(tmp, "release")
            evidence = produce_release(pkg)
            self.assertTrue(evidence["passed"])
            self.assertEqual(evidence["final_state"], "offstage")
            self.assertEqual(evidence["frame_count"], 510)
            self.assertEqual(
                [s["state"] for s in evidence["state_path"]],
                ["waiting", "walking_in", "presenting", "exiting", "offstage"],
            )
            names = {"manifest.json", "bundle.json", "runtime_config.json",
                     "evidence.json"}
            self.assertEqual(set(os.listdir(pkg)), names)
            with open(os.path.join(pkg, "evidence.json"), encoding="utf-8") as fh:
                on_disk = json.load(fh)
            self.assertEqual(on_disk["playback_frames_sha256"],
                             evidence["playback_frames_sha256"])
            manifest, bundle, config = load_package(pkg)
            self.assertEqual(len(bundle.clips), 5)

    def test_package_build_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a")
            b = os.path.join(tmp, "b")
            ea = produce_release(a)
            eb = produce_release(b)
            self.assertEqual(ea["playback_frames_sha256"],
                             eb["playback_frames_sha256"])
            self.assertEqual(ea["bundle_sha256"], eb["bundle_sha256"])
            self.assertEqual(ea["runtime_config_sha256"],
                             eb["runtime_config_sha256"])

    def test_startup_script_runs_without_editor_and_writes_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "evidence")
            proc = subprocess.run(
                ["bash", os.path.join(ROOT, "start_actor.sh"), "--out", out],
                cwd=tmp, capture_output=True, text=True,
                env={**os.environ, "PYTHON": sys.executable},
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue(os.path.isfile(os.path.join(out, "evidence.json")),
                            "启动后必须留下真实回放证据")

    def test_startup_script_fails_closed_on_missing_resource(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = os.path.join(tmp, "release")
            produce_release(pkg)
            os.remove(os.path.join(pkg, "bundle.json"))
            out = os.path.join(tmp, "evidence")
            proc = subprocess.run(
                ["bash", os.path.join(ROOT, "start_actor.sh"),
                 "--package", pkg, "--out", out],
                cwd=tmp, capture_output=True, text=True,
                env={**os.environ, "PYTHON": sys.executable},
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertFalse(os.path.exists(
                os.path.join(out, "evidence.json")))


class AcceptanceStopsTests(unittest.TestCase):
    def _bundle_with_missing_clip(self):
        bundle = build_bundle()
        states = list(bundle.graph.states.values())
        states.append(State("phantom", "no_such_clip"))
        graph = StateGraph(states, bundle.graph.transitions,
                           bundle.graph.initial_state)
        return Bundle(bundle.skeleton, list(bundle.clips.values()), graph)

    def test_missing_action_stops_acceptance_without_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out")
            os.makedirs(out)
            stale = os.path.join(out, "evidence.json")
            with open(stale, "w", encoding="utf-8") as fh:
                fh.write('{"passed": true}')
            with self.assertRaises(ReleaseError) as ctx:
                run_selfcheck(self._bundle_with_missing_clip(),
                              default_runtime_config(), out)
            self.assertIn("不存在的动作片段", str(ctx.exception))
            self.assertFalse(os.path.exists(stale), "失败后旧证据必须清除")

    def test_parameter_type_conflict_stops_before_frames(self):
        config = default_runtime_config()
        step = next(s for s in config["sample"]["script"]
                    if s.get("cmd") == "set" and s.get("name") == "entry_cue")
        step["value"] = "true"                     # bool 声明收到 str
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out")
            with self.assertRaises(ReleaseError) as ctx:
                run_selfcheck(build_bundle(), config, out)
            self.assertIn("参数类型冲突", str(ctx.exception))
            self.assertFalse(os.path.exists(
                os.path.join(out, "evidence.json")))

    def test_undeclared_parameter_is_rejected(self):
        config = default_runtime_config()
        config["sample"]["script"].append(
            {"at": 0, "cmd": "set", "name": "mystery", "value": True})
        with self.assertRaises(ReleaseError) as ctx:
            run_selfcheck(build_bundle(), config,
                          tempfile.mkdtemp())
        self.assertIn("未声明参数", str(ctx.exception))

    def test_missing_resource_stops_startup(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = os.path.join(tmp, "release")
            produce_release(pkg)
            os.remove(os.path.join(pkg, "bundle.json"))
            with self.assertRaises(ReleaseError) as ctx:
                start_package(pkg, os.path.join(tmp, "out"))
            self.assertIn("资源缺失", str(ctx.exception))
            self.assertFalse(os.path.exists(
                os.path.join(tmp, "out", "evidence.json")))

    def test_resource_hash_mismatch_stops_startup(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = os.path.join(tmp, "release")
            produce_release(pkg)
            with open(os.path.join(pkg, "bundle.json"), encoding="utf-8") as fh:
                data = json.load(fh)
            data["graph"]["states"][0]["clip_name"] = "tampered"
            with open(os.path.join(pkg, "bundle.json"), "w",
                      encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False)
            with self.assertRaises(ReleaseError) as ctx:
                start_package(pkg, os.path.join(tmp, "out"))
            self.assertIn("资源内容与清单不一致", str(ctx.exception))

    def test_unknown_event_subscription_is_rejected(self):
        config = default_runtime_config()
        config["event_subscriptions"].append(
            {"name": "ghost_cue", "min_count": 1, "max_count": 1})
        with self.assertRaises(ReleaseError) as ctx:
            run_selfcheck(build_bundle(), config, tempfile.mkdtemp())
        self.assertIn("不可能触发的事件", str(ctx.exception))

    def test_event_count_mismatch_stops_acceptance(self):
        config = default_runtime_config()
        foot = next(s for s in config["event_subscriptions"]
                    if s["name"] == "footstep")
        foot["min_count"] = foot["max_count"] = 9      # 实际只有 8 次
        with self.assertRaises(ReleaseError) as ctx:
            run_selfcheck(build_bundle(), config, tempfile.mkdtemp())
        self.assertIn("订阅数量不符", str(ctx.exception))

    def test_state_trajectory_mismatch_stops_acceptance(self):
        config = default_runtime_config()
        config["expected"]["state_path"] = [
            "waiting", "presenting", "walking_in", "exiting", "offstage"]
        with self.assertRaises(ReleaseError) as ctx:
            run_selfcheck(build_bundle(), config, tempfile.mkdtemp())
        self.assertIn("状态轨迹不符", str(ctx.exception))

    def test_parameter_default_is_not_used_to_mask_errors(self):
        config = default_runtime_config()
        config["parameters"][0]["default"] = True
        with self.assertRaises(ReleaseError) as ctx:
            run_selfcheck(build_bundle(), config, tempfile.mkdtemp())
        self.assertIn("未支持字段", str(ctx.exception))

    def test_broken_release_build_leaves_no_artifacts(self):
        config = default_runtime_config()
        config["parameters"][0]["type"] = "number"   # 未知类型
        with tempfile.TemporaryDirectory() as tmp:
            pkg = os.path.join(tmp, "release")
            with self.assertRaises(ReleaseError):
                produce_release(pkg, config=config)
            produced = {n for n in os.listdir(pkg)
                        if n in {"manifest.json", "bundle.json",
                                 "runtime_config.json", "evidence.json"}}
            self.assertEqual(produced, set(), "失败不得发布任何产物")


class InterruptingTransitionTests(unittest.TestCase):
    def test_interrupting_condition_is_deferred_then_applied(self):
        # a->b 过渡进行中把 b->c 的条件置真：不得抢占，也不得丢弃。
        skel = build_bundle().skeleton
        idle, gesture = build_idle_clip(), build_gesture_clip()
        states = [State("a", "idle"), State("b", "gesture"),
                  State("c", "idle")]
        transitions = [
            Transition("a", "b", [Condition("go", "==", True)], 0.2,
                       phase_mode="restart"),
            Transition("b", "c", [Condition("go2", "==", True)], 0.2,
                       phase_mode="restart"),
        ]
        graph = StateGraph(states, transitions, "a")
        machine = ActorMachine(Bundle(skel, [idle, gesture], graph))

        machine.set_variable("go", True)
        machine.step(DT)
        self.assertIsNotNone(machine.transition)
        self.assertEqual(machine.transition[0].target, "b")
        machine.set_variable("go2", True)
        machine.step(DT)
        # 过渡仍保持 a->b，未被中途打断
        self.assertEqual(machine.transition[0].target, "b")

        for _ in range(60):
            machine.step(DT)
            if machine.current_state == "c" and machine.transition is None:
                break
        self.assertEqual(machine.current_state, "c",
                         "被打断的转换条件必须在过渡结束后落实")
        self.assertIsNone(machine.transition)


if __name__ == "__main__":
    unittest.main(verbosity=2)
