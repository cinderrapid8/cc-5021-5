"""命令行入口：构建 -> 固定脚本回放 -> 保存状态图/回放数据 -> 自动核对。"""

import argparse
import json
import os
import sys

from .authoring import SCRIPT_SECONDS, build_bundle, build_script
from .checks import assert_no_pose_jumps, compressed_state_path
from .graph import Bundle
from .machine import ActorMachine
from .replay import ReplayRunner


def run_demo(out_dir, frame_rate=30.0, seek_policy=ActorMachine.SEEK_SUPPRESS):
    os.makedirs(out_dir, exist_ok=True)
    dt = 1.0 / frame_rate

    bundle = build_bundle()
    machine = ActorMachine(bundle, seek_policy=seek_policy)
    runner = ReplayRunner(machine, dt=dt)
    script = build_script(dt)
    frame_count = round(SCRIPT_SECONDS * frame_rate)
    frames = runner.run(script, frame_count)

    graph_path = os.path.join(out_dir, "state_graph.json")
    replay_path = os.path.join(out_dir, "replay.json")
    viewer_path = os.path.join(out_dir, "viewer.html")
    with open(graph_path, "w", encoding="utf-8") as fh:
        json.dump(bundle.to_dict(), fh, ensure_ascii=False, indent=1)
    runner.export_json(replay_path)
    from .viewer import build_viewer
    build_viewer(runner.export_dict(), viewer_path)

    report = _summarize(runner)
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    return runner, report, graph_path, replay_path


def _summarize(runner):
    export = runner.export_dict()
    frames = export["frames"]
    worst, where = assert_no_pose_jumps(frames)
    return {
        "frame_count": len(frames),
        "dt": export["dt"],
        "seek_policy": export["seek_policy"],
        "state_path": compressed_state_path(frames),
        "event_total": len(export["all_events"]),
        "event_by_name": {
            name: sum(1 for e in export["all_events"] if e["name"] == name)
            for name in sorted({e["name"] for e in export["all_events"]})
        },
        "worst_frame_jump": {"distance": round(worst, 6), "frame": where},
    }


def verify_roundtrip(graph_path):
    """从保存的状态图整包重建并回放，核对与原运行完全一致。"""
    with open(graph_path, "r", encoding="utf-8") as fh:
        bundle = Bundle.from_dict(json.load(fh))
    dt = 1.0 / 30.0
    runner = ReplayRunner(ActorMachine(bundle), dt=dt)
    frames = runner.run(build_script(dt), round(SCRIPT_SECONDS * 30.0))
    return runner, _summarize(runner)


def main(argv=None):
    parser = argparse.ArgumentParser(description="虚拟舞台演员动作控制系统")
    parser.add_argument("command",
                        choices=["demo", "verify", "release", "selfcheck"],
                        help="demo: 构建并回放；verify: 从已保存状态图重建核对；"
                             "release: 生成发布包并自检；selfcheck: 自检现有发布包")
    parser.add_argument("--out", default="out", help="输出目录")
    parser.add_argument("--package", default="release", help="发布包目录")
    parser.add_argument("--seek-policy", choices=[ActorMachine.SEEK_SUPPRESS,
                                                   ActorMachine.SEEK_REPLAY],
                        default=ActorMachine.SEEK_SUPPRESS)
    args = parser.parse_args(argv)

    if args.command == "release":
        from .release import produce_release
        evidence = produce_release(args.package)
        print("== 发布包构建与自检完成 ==")
        print(json.dumps(evidence, ensure_ascii=False, indent=2))
        print(f"发布包目录: {os.path.abspath(args.package)}")
        return 0

    if args.command == "selfcheck":
        from .release import ReleaseError
        from .runtime import start_package
        try:
            evidence = start_package(args.package, args.out)
        except ReleaseError as exc:
            print(f"自检失败，拒绝采用产物:\n{exc}", file=sys.stderr)
            return 2
        print("== 发布包自检通过 ==")
        print(json.dumps(evidence, ensure_ascii=False, indent=2))
        return 0

    if args.command == "demo":
        runner, report, graph_path, replay_path = run_demo(
            args.out, seek_policy=args.seek_policy)
        print("== 固定脚本回放完成 ==")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"状态图整包: {graph_path}")
        print(f"回放数据:   {replay_path}")
        print(f"回放界面:   {os.path.join(args.out, 'viewer.html')}")
        return 0

    graph_path = os.path.join(args.out, "state_graph.json")
    _, report2 = verify_roundtrip(graph_path)
    print("== 状态图重新加载回放 ==")
    print(json.dumps(report2, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
