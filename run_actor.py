#!/usr/bin/env python3
"""发布包启动入口：加载 release/ 下的运行配置并执行旧样例自检。

用法：
    ./run_actor.py [--package DIR] [--out DIR]

退出码 0 表示资源、参数、事件订阅、中断转换、状态轨迹与播放帧全部通过；
非 0 表示验收失败，且不会留下证据文件。
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from stageactor.release import ReleaseError  # noqa: E402
from stageactor.runtime import start_package  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description="虚拟舞台演员发布包自检")
    parser.add_argument("--package", default=os.path.join(ROOT, "release"),
                        help="发布包目录")
    parser.add_argument("--out", default=None,
                        help="证据输出目录（默认与发布包同目录）")
    args = parser.parse_args(argv)

    try:
        evidence = start_package(args.package, args.out)
    except ReleaseError as exc:
        print(f"自检失败，拒绝采用产物:\n{exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "passed": evidence["passed"],
        "final_state": evidence["final_state"],
        "frame_count": evidence["frame_count"],
        "state_path": evidence["state_path"],
        "worst_frame_jump": evidence["worst_frame_jump"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
