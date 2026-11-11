"""动作运行器发布：打包运行器依赖与参数配置，并在无编辑器环境下自检。

发布包由三部分组成：
  bundle.json          动画依赖（骨骼 / 片段 / 状态图）
  runtime_config.json  参数配置、事件订阅、样例脚本与验收基线
  manifest.json        资源引用与内容摘要、必需检查项、证据文件名

自检不接受“窗口能打开”这种弱证据：它必须用旧样例真正回放到底，覆盖暂停 /
变速 / 前后跳转等打断与全部状态转换，并核对资源引用、参数类型、事件订阅、
状态轨迹和逐帧播放数据。任何一项失败都抛 ReleaseError，且不写出证据文件。
"""

import hashlib
import json
import os
import shutil
import tempfile

from .authoring import SCRIPT_SECONDS, build_bundle, build_script
from .checks import assert_no_pose_jumps, compressed_state_path
from .graph import Bundle
from .machine import ActorMachine
from .replay import ReplayRunner

DEFAULT_FRAME_RATE = 30.0
RELEASE_NAME = "stage-actor-runner"
RELEASE_VERSION = "1.0.0"
CHECK_NAMES = (
    "resource_references", "parameter_types", "event_subscriptions",
    "interruption_transitions", "state_trajectory", "playback_frames",
)

_PARAM_TYPE_CHECKERS = {
    "bool": lambda v: isinstance(v, bool),
    "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "float": lambda v: (isinstance(v, (int, float))
                        and not isinstance(v, bool)),
    "str": lambda v: isinstance(v, str),
}
_COMMANDS = ("set", "pause", "resume", "rate", "seek")
_PHASE_MODES = ("restart", "phase_match", "resume")


class ReleaseError(RuntimeError):
    """发布或自检失败；调用方必须拒绝采用该包的任何产物。"""


# ============================================================ 严格读取（无默认值）
def _require(mapping, key, where):
    if not isinstance(mapping, dict):
        raise ReleaseError(f"{where} 必须是对象")
    if key not in mapping:
        raise ReleaseError(f"缺少必填配置项: {where}.{key}")
    return mapping[key]


def _type_name(value):
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    return type(value).__name__


def canonical_json(obj):
    """稳定序列化，用于摘要与跨环境一致性比较。"""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def sha256_of(obj):
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


# ============================================================ 资源引用校验
def validate_bundle(bundle):
    """核对动作、状态与骨骼的引用完整性；缺失动作直接判失败。"""
    errors = []
    skel = bundle.skeleton
    names = skel.names
    if not names:
        errors.append("骨骼为空")
    if len(set(names)) != len(names):
        errors.append("骨骼名称重复")
    for i, bone in enumerate(skel.bones):
        if bone.parent < -1:
            errors.append(f"骨骼 {bone.name} 的父索引非法: {bone.parent}")
        elif bone.parent >= i:
            errors.append(f"骨骼 {bone.name} 的父索引未指向更早的骨骼")
        if bone.length < 0:
            errors.append(f"骨骼 {bone.name} 长度为负")

    if not bundle.clips:
        errors.append("动画片段为空")
    for name, clip in bundle.clips.items():
        if name != clip.name:
            errors.append(f"片段字典键 {name} 与片段名 {clip.name} 不一致")
        if clip.duration <= 0:
            errors.append(f"片段 {name} 时长必须为正: {clip.duration}")
        if not clip.keyframes:
            errors.append(f"片段 {name} 缺少关键帧")
        for k in clip.keyframes:
            if k.time < 0 or k.time > clip.duration:
                errors.append(
                    f"片段 {name} 关键帧时间越界: {k.time}")
        if clip.loop and len(clip.keyframes) > 1:
            if clip.keyframes[-1].time <= clip.keyframes[0].time:
                errors.append(f"循环片段 {name} 的时间跨度为零")
            if clip.tail_blend > clip.duration:
                errors.append(f"循环片段 {name} 的尾部混合窗口超过时长")
        for ev in clip.events:
            if ev.time < 0 or ev.time > clip.duration:
                errors.append(f"片段 {name} 事件 {ev.name} 时间越界: {ev.time}")

    states = bundle.graph.states
    if bundle.graph.initial_state not in states:
        errors.append("初始状态不在状态表中")
    for st in states.values():
        if st.clip_name not in bundle.clips:
            errors.append(
                f"状态 {st.name} 引用了不存在的动作片段 {st.clip_name}")
        elif st.auto_finish and bundle.clips[st.clip_name].loop:
            errors.append(
                f"状态 {st.name} 配置 auto_finish 但片段 {st.clip_name} 是循环的")
    for tr in bundle.graph.transitions:
        if tr.source not in states:
            errors.append(f"过渡源状态不存在: {tr.source}")
        if tr.target not in states:
            errors.append(f"过渡目标状态不存在: {tr.target}")
        if tr.duration <= 0:
            errors.append(
                f"过渡 {tr.source}->{tr.target} 时长必须为正: {tr.duration}")
        if tr.phase_mode not in _PHASE_MODES:
            errors.append(
                f"过渡 {tr.source}->{tr.target} 相位规则未知: {tr.phase_mode}")

    if errors:
        raise ReleaseError("资源引用校验失败:\n  " + "\n  ".join(errors))


# ============================================================ 参数配置校验
def validate_parameter_schema(schema):
    """参数配置只允许显式声明；拒绝 default 之类会遮错的回退值。"""
    errors = []
    if not isinstance(schema, list):
        raise ReleaseError("参数配置必须是列表")
    seen = set()
    for i, entry in enumerate(schema):
        where = f"parameters[{i}]"
        if not isinstance(entry, dict):
            errors.append(f"{where} 必须是对象")
            continue
        extra = set(entry) - {"name", "type", "required"}
        if extra:
            errors.append(f"{where} 含未支持字段: {sorted(extra)}")
        name = _require(entry, "name", where)
        type_name = _require(entry, "type", where)
        required = _require(entry, "required", where)
        if not isinstance(name, str) or not name:
            errors.append(f"{where}.name 必须是非空字符串")
        elif name.startswith("__"):
            errors.append(f"{where}.name 不得占用内部变量名 {name}")
        if type_name not in _PARAM_TYPE_CHECKERS:
            errors.append(f"{where}.type 未知: {type_name!r}")
        if type_name == "bool" and "default" in entry:
            errors.append(f"{where} 不得以默认值遮错")
        if not isinstance(required, bool):
            errors.append(f"{where}.required 必须是 bool")
        if name in seen:
            errors.append(f"参数 {name} 重复声明")
        seen.add(name)
    if errors:
        raise ReleaseError("参数配置校验失败:\n  " + "\n  ".join(errors))


def _check_value_type(name, type_name, value):
    checker = _PARAM_TYPE_CHECKERS.get(type_name)
    if checker is None:
        raise ReleaseError(f"参数 {name} 的类型未知: {type_name!r}")
    if not checker(value):
        raise ReleaseError(
            f"参数类型冲突: {name} 声明为 {type_name}，实际为 "
            f"{_type_name(value)} ({value!r})")


def validate_sample_script(script, schema, frame_count):
    """运行前静态校验脚本：命令、帧号、参数声明与类型都必须合法。

    参数类型冲突在回放开始前就停止验收，不产生任何帧产物。
    """
    if not isinstance(script, list):
        raise ReleaseError("样例脚本必须是步骤列表")
    declared = {p["name"]: p for p in schema}
    supplied = set()
    for i, step in enumerate(script):
        where = f"script[{i}]"
        if not isinstance(step, dict):
            raise ReleaseError(f"{where} 必须是对象")
        at = _require(step, "at", where)
        cmd = _require(step, "cmd", where)
        if not isinstance(at, int) or isinstance(at, bool):
            raise ReleaseError(f"{where}.at 必须是整数帧号")
        if at < 0 or at >= frame_count:
            raise ReleaseError(f"{where}.at 超出回放帧范围: {at}")
        if cmd not in _COMMANDS:
            raise ReleaseError(f"{where}.cmd 未知: {cmd!r}")
        if cmd == "set":
            name = _require(step, "name", where)
            value = _require(step, "value", where)
            if name not in declared:
                raise ReleaseError(f"{where} 设置了未声明参数: {name}")
            _check_value_type(name, declared[name]["type"], value)
            supplied.add(name)
        elif cmd == "rate":
            value = _require(step, "value", where)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ReleaseError(f"{where}.value 必须是数值速率")
        elif cmd == "seek":
            value = _require(step, "time", where)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ReleaseError(f"{where}.time 必须是数值时间")
    missing = {name for name, p in declared.items()
               if p["required"] and name not in supplied}
    if missing:
        raise ReleaseError("缺少必填参数输入: " + ", ".join(sorted(missing)))
    return supplied


# ============================================================ 事件订阅校验
def known_event_names(bundle):
    names = set()
    for clip in bundle.clips.values():
        for ev in clip.events:
            names.add(ev.name)
    for tr in bundle.graph.transitions:
        if tr.cue:
            names.add(tr.cue)
    return names


def validate_event_subscriptions(subscriptions, bundle):
    errors = []
    if not isinstance(subscriptions, list):
        raise ReleaseError("事件订阅必须是列表")
    known = known_event_names(bundle)
    seen = set()
    for i, sub in enumerate(subscriptions):
        where = f"event_subscriptions[{i}]"
        if not isinstance(sub, dict):
            errors.append(f"{where} 必须是对象")
            continue
        extra = set(sub) - {"name", "min_count", "max_count"}
        if extra:
            errors.append(f"{where} 含未支持字段: {sorted(extra)}")
        name = _require(sub, "name", where)
        lo = _require(sub, "min_count", where)
        hi = _require(sub, "max_count", where)
        if name not in known:
            errors.append(f"{where} 订阅了不可能触发的事件: {name}")
        if not isinstance(lo, int) or isinstance(lo, bool) or lo < 0:
            errors.append(f"{where}.min_count 必须是非负整数")
        if not isinstance(hi, int) or isinstance(hi, bool) or hi < lo:
            errors.append(f"{where}.max_count 必须是不小于 min_count 的整数")
        if name in seen:
            errors.append(f"事件 {name} 重复订阅")
        seen.add(name)
    if errors:
        raise ReleaseError("事件订阅校验失败:\n  " + "\n  ".join(errors))


# ============================================================ 配置校验
def validate_runtime_config(config, bundle):
    _require(config, "format", "config")
    if config["format"] != 1:
        raise ReleaseError(f"不支持的配置格式: {config['format']}")
    policy = _require(config, "seek_policy", "config")
    if policy not in (ActorMachine.SEEK_SUPPRESS, ActorMachine.SEEK_REPLAY):
        raise ReleaseError(f"未知 seek 政策: {policy!r}")
    sample = _require(config, "sample", "config")
    rate = _require(sample, "frame_rate", "config.sample")
    seconds = _require(sample, "seconds", "config.sample")
    script = _require(sample, "script", "config.sample")
    if not isinstance(rate, (int, float)) or isinstance(rate, bool) or rate <= 0:
        raise ReleaseError("config.sample.frame_rate 必须为正数")
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or seconds <= 0:
        raise ReleaseError("config.sample.seconds 必须为正数")
    frame_count = round(seconds * rate)

    schema = _require(config, "parameters", "config")
    validate_parameter_schema(schema)
    subscriptions = _require(config, "event_subscriptions", "config")
    validate_event_subscriptions(subscriptions, bundle)
    expected = _require(config, "expected", "config")
    for key in ("frame_count", "state_path", "final_state",
                "max_frame_jump", "event_total"):
        _require(expected, key, "config.expected")
    if expected["frame_count"] != frame_count:
        raise ReleaseError(
            "config.sample 与 expected.frame_count 不一致: "
            f"{frame_count} != {expected['frame_count']}")
    states = bundle.graph.states
    for name in expected["state_path"]:
        if name not in states:
            raise ReleaseError(f"期望状态路径含未知状态: {name}")
    if expected["final_state"] not in states:
        raise ReleaseError(f"期望最终状态未知: {expected['final_state']}")
    validate_sample_script(script, schema, frame_count)
    return frame_count


# ============================================================ 真实回放与验收
def _event_counts(export):
    counts = {}
    for ev in export["all_events"]:
        counts[ev["name"]] = counts.get(ev["name"], 0) + 1
    return counts


def _check_event_subscriptions(export, subscriptions):
    counts = _event_counts(export)
    details = {}
    errors = []
    for sub in subscriptions:
        got = counts.get(sub["name"], 0)
        details[sub["name"]] = got
        if not (sub["min_count"] <= got <= sub["max_count"]):
            errors.append(
                f"事件 {sub['name']} 订阅数量不符: 实际 {got}，期望 "
                f"[{sub['min_count']}, {sub['max_count']}]")
    if errors:
        raise ReleaseError("事件订阅运行校验失败:\n  " + "\n  ".join(errors))
    return details


def _paused_blocks(frames):
    blocks = []
    current = None
    for fr in frames:
        if fr["paused"]:
            if current is None:
                current = [fr["frame"], fr["frame"], fr["time"]]
            else:
                current[1] = fr["frame"]
        elif current is not None:
            blocks.append(tuple(current))
            current = None
    if current is not None:
        blocks.append(tuple(current))
    return blocks


def _transition_windows(frames):
    windows = []
    current = None
    for fr in frames:
        tgt = fr["transition_target"]
        if tgt is not None:
            if current is None:
                current = {"start_frame": fr["frame"], "target": tgt,
                           "progress": []}
            current["progress"].append(fr["transition_progress"])
            current["end_frame"] = fr["frame"]
        elif current is not None:
            windows.append(current)
            current = None
    if current is not None:
        windows.append(current)
    return windows


def _check_interruptions(frames, expected):
    """中断与转换必须真实发生并完整收束：暂停冻结时钟、跳转被记录、
    过渡窗口单调推进且数量覆盖整条状态路径。"""
    errors = []
    paused = _paused_blocks(frames)
    if not paused:
        errors.append("旧样例未包含暂停打断")
    for first, last, frozen in paused:
        span = [f for f in frames if first <= f["frame"] <= last]
        if any(round(f["time"], 9) != round(frozen, 9) for f in span):
            errors.append(f"暂停区间 {first}-{last} 内主时钟仍在推进")

    seeked = [f["frame"] for f in frames if f["seeked"]]
    if not seeked:
        errors.append("旧样例未包含时间跳转打断")

    rates = {f["rate"] for f in frames}
    if not any(abs(r - 1.0) > 1e-9 for r in rates):
        errors.append("旧样例未包含变速打断")

    windows = _transition_windows(frames)
    if frames and frames[-1]["transition_target"] is not None:
        errors.append("回放结束时仍有未收束的过渡")
    expected_count = max(0, len(expected["state_path"]) - 1)
    if len(windows) != expected_count:
        errors.append(
            f"过渡数量不符: 实际 {len(windows)}，期望 {expected_count}")
    for win in windows:
        progress = win["progress"]
        if progress and progress[0] < 0.0:
            errors.append(f"过渡 {win['target']} 进度为负")
        for a, b in zip(progress, progress[1:]):
            if b < a:
                errors.append(f"过渡 {win['target']} 进度回退: {a} -> {b}")
    if errors:
        raise ReleaseError("中断/转换检查失败:\n  " + "\n  ".join(errors))
    return {
        "paused_blocks": [{"from": p[0], "to": p[1]} for p in paused],
        "seek_frames": seeked,
        "rate_values": sorted(rates),
        "transition_count": len(windows),
        "transition_targets": [w["target"] for w in windows],
    }


def _check_trajectory(frames, expected):
    path = compressed_state_path(frames)
    names = [p[0] for p in path]
    if names != list(expected["state_path"]):
        raise ReleaseError(
            "状态轨迹不符:\n  实际 " + " -> ".join(names) +
            "\n  期望 " + " -> ".join(expected["state_path"]))
    if not frames or frames[-1]["state"] != expected["final_state"]:
        raise ReleaseError(
            f"最终状态不符: {frames[-1]['state'] if frames else None} != "
            f"{expected['final_state']}")
    return [{"state": s, "frame": f} for s, f in path]


def _check_playback_frames(frames, export, expected, skeleton):
    errors = []
    if len(frames) != expected["frame_count"]:
        errors.append(
            f"播放帧数不符: 实际 {len(frames)}，期望 {expected['frame_count']}")
    bone_count = len(skeleton.bones)
    for fr in frames:
        if len(fr["joint_positions"]) != bone_count:
            errors.append(
                f"帧 {fr['frame']} 关节数 {len(fr['joint_positions'])} != {bone_count}")
            break
        if abs(sum(fr["weights"].values()) - 1.0) > 1e-5:
            errors.append(f"帧 {fr['frame']} 混合权重和不为 1")
            break
    worst, where = assert_no_pose_jumps(frames)
    if worst > expected["max_frame_jump"]:
        errors.append(
            f"逐帧最大位移 {worst:.6f} 超过阈值 {expected['max_frame_jump']}")
    total = len(export["all_events"])
    if total != expected["event_total"]:
        errors.append(f"事件总数不符: 实际 {total}，期望 {expected['event_total']}")
    if errors:
        raise ReleaseError("播放帧校验失败:\n  " + "\n  ".join(errors))
    return {"frame_count": len(frames),
            "worst_frame_jump": {"distance": round(worst, 6), "frame": where},
            "event_total": total}


def run_selfcheck(bundle, config, out_dir, manifest=None):
    """在无编辑器环境下执行完整验收；只有全部通过才写出证据。"""
    os.makedirs(out_dir, exist_ok=True)
    evidence_name = (manifest or {}).get("evidence", "evidence.json")
    evidence_path = os.path.join(out_dir, evidence_name)
    if os.path.exists(evidence_path):
        os.remove(evidence_path)          # 旧证据不得掩盖本次失败

    validate_bundle(bundle)
    frame_count = validate_runtime_config(config, bundle)

    sample = config["sample"]
    dt = 1.0 / float(sample["frame_rate"])
    machine = ActorMachine(bundle, seek_policy=config["seek_policy"])
    runner = ReplayRunner(machine, dt=dt)
    # validate_runtime_config 已在生成任何帧之前完成参数类型校验。
    runner.run(sample["script"], frame_count)
    export = runner.export_dict()
    frames = export["frames"]

    subscription_counts = _check_event_subscriptions(
        export, config["event_subscriptions"])
    interruptions = _check_interruptions(frames, config["expected"])
    state_path = _check_trajectory(frames, config["expected"])
    playback = _check_playback_frames(
        frames, export, config["expected"], bundle.skeleton)

    checks = [
        {"name": "resource_references", "ok": True,
         "detail": f"{len(bundle.clips)} 个动作 / {len(bundle.graph.states)} 个状态"},
        {"name": "parameter_types", "ok": True,
         "detail": f"{len(config['parameters'])} 个声明参数"},
        {"name": "event_subscriptions", "ok": True,
         "detail": subscription_counts},
        {"name": "interruption_transitions", "ok": True, "detail": interruptions},
        {"name": "state_trajectory", "ok": True, "detail": state_path},
        {"name": "playback_frames", "ok": True, "detail": playback},
    ]

    evidence = {
        "release": (manifest or {}).get("release", RELEASE_NAME),
        "version": (manifest or {}).get("version", RELEASE_VERSION),
        "passed": True,
        "seek_policy": export["seek_policy"],
        "dt": export["dt"],
        "frame_count": playback["frame_count"],
        "state_path": state_path,
        "final_state": frames[-1]["state"],
        "event_total": playback["event_total"],
        "event_counts": _event_counts(export),
        "subscription_counts": subscription_counts,
        "interruptions": interruptions,
        "worst_frame_jump": playback["worst_frame_jump"],
        "playback_frames_sha256": sha256_of(frames),
        "bundle_sha256": (manifest or {}).get(
            "resources", {}).get("bundle", {}).get("sha256"),
        "runtime_config_sha256": sha256_of(config),
        "checks": checks,
    }
    _write_json_atomic(evidence_path, evidence)
    return evidence


# ============================================================ 发布包构建
def default_runtime_config():
    dt = 1.0 / DEFAULT_FRAME_RATE
    return {
        "format": 1,
        "seek_policy": ActorMachine.SEEK_SUPPRESS,
        "sample": {
            "frame_rate": DEFAULT_FRAME_RATE,
            "seconds": SCRIPT_SECONDS,
            "script": build_script(dt),
        },
        "parameters": [
            {"name": "entry_cue", "type": "bool", "required": True},
            {"name": "exit_cue", "type": "bool", "required": True},
        ],
        "event_subscriptions": [
            {"name": "footstep", "min_count": 8, "max_count": 8},
            {"name": "gesture_cue", "min_count": 8, "max_count": 8},
            {"name": "breath_cue", "min_count": 1, "max_count": 1},
            {"name": "arrival_cue", "min_count": 1, "max_count": 1},
            {"name": "bow_cue", "min_count": 1, "max_count": 1},
            {"name": "entry_signal", "min_count": 1, "max_count": 1},
            {"name": "arrival_signal", "min_count": 1, "max_count": 1},
            {"name": "exit_signal", "min_count": 1, "max_count": 1},
        ],
        "expected": {
            "frame_count": round(SCRIPT_SECONDS * DEFAULT_FRAME_RATE),
            "state_path": ["waiting", "walking_in", "presenting",
                           "exiting", "offstage"],
            "final_state": "offstage",
            "max_frame_jump": 0.6,
            "event_total": 22,
        },
    }


def _manifest_for(bundle_dict, config_dict):
    return {
        "format": 1,
        "release": RELEASE_NAME,
        "version": RELEASE_VERSION,
        "runner": {"entry": "run_actor.py",
                   "module": "stageactor.runtime",
                   "requires": "python3"},
        "resources": {
            "bundle": {
                "path": "bundle.json",
                "sha256": sha256_of(bundle_dict),
                "clips": sorted(bundle_dict_clips(bundle_dict)),
                "states": sorted(s["name"] for s in bundle_dict["graph"]["states"]),
            },
            "runtime_config": {
                "path": "runtime_config.json",
                "sha256": sha256_of(config_dict),
            },
        },
        "required_checks": list(CHECK_NAMES),
        "evidence": "evidence.json",
    }


def bundle_dict_clips(bundle_dict):
    return [c["name"] for c in bundle_dict["clips"]]


def _discard(path):
    if os.path.exists(path):
        os.remove(path)


def _write_json_atomic(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1, sort_keys=True)
    os.replace(tmp, path)


def produce_release(package_dir, config=None, bundle=None):
    """生成发布包并立即自检；自检不过则不发布任何文件。

    先在临时目录构建并跑完旧样例，全部通过后才把清单、资源与证据写入
    package_dir，避免失败产物被后续流程误用。
    """
    bundle = bundle if bundle is not None else build_bundle()
    config = config if config is not None else default_runtime_config()
    # 失败不得沿用上一次的证据。
    os.makedirs(package_dir, exist_ok=True)
    _discard(os.path.join(package_dir, "evidence.json"))
    bundle_dict = bundle.to_dict()
    validate_bundle(bundle)
    validate_runtime_config(config, bundle)
    manifest = _manifest_for(bundle_dict, config)

    with tempfile.TemporaryDirectory() as tmp:
        _write_json_atomic(os.path.join(tmp, "bundle.json"), bundle_dict)
        _write_json_atomic(os.path.join(tmp, "runtime_config.json"), config)
        _write_json_atomic(os.path.join(tmp, "manifest.json"), manifest)
        evidence = run_selfcheck(bundle, config, tmp, manifest=manifest)
        published = ("manifest.json", "bundle.json",
                     "runtime_config.json", manifest["evidence"])
        os.makedirs(package_dir, exist_ok=True)
        for name in published:
            shutil.copyfile(os.path.join(tmp, name),
                            os.path.join(package_dir, name))
    return evidence


# ============================================================ 发布包读取
def load_manifest(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except FileNotFoundError:
        raise ReleaseError(f"发布清单缺失: {path}")
    except json.JSONDecodeError as exc:
        raise ReleaseError(f"发布清单不是合法 JSON: {exc}")
    _require(manifest, "format", "manifest")
    if manifest["format"] != 1:
        raise ReleaseError(f"不支持的发布清单格式: {manifest['format']}")
    for key in ("release", "version", "runner", "resources",
                "required_checks", "evidence"):
        _require(manifest, key, "manifest")
    missing = set(CHECK_NAMES) - set(manifest["required_checks"])
    if missing:
        raise ReleaseError("发布清单缺少必需检查项: " + ", ".join(sorted(missing)))
    return manifest


def _load_resource(package_dir, resource, label):
    path = _require(resource, "path", f"manifest.resources.{label}")
    expected_hash = _require(resource, "sha256", f"manifest.resources.{label}")
    full = os.path.join(package_dir, path)
    if not os.path.isfile(full):
        raise ReleaseError(f"资源缺失: {label} -> {path}")
    try:
        with open(full, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ReleaseError(f"资源不是合法 JSON: {path}: {exc}")
    actual = sha256_of(payload)
    if actual != expected_hash:
        raise ReleaseError(
            f"资源内容与清单不一致: {label} ({actual} != {expected_hash})")
    return payload


def load_package(package_dir):
    package_dir = os.path.abspath(package_dir)
    manifest = load_manifest(os.path.join(package_dir, "manifest.json"))
    resources = _require(manifest, "resources", "manifest")
    bundle_dict = _load_resource(package_dir,
                                 _require(resources, "bundle",
                                          "manifest.resources"), "bundle")
    config = _load_resource(
        package_dir, _require(resources, "runtime_config",
                              "manifest.resources"), "runtime_config")
    bundle = Bundle.from_dict(bundle_dict)
    return manifest, bundle, config
