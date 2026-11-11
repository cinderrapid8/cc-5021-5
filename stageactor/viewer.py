"""把回放数据注入 HTML 模板，生成可离线打开的单文件回放界面。"""

import json
import os

_TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "viewer_template.html")
_MARKER = "// 回放数据由 viewer.py 生成为 window.REPLAY"


def build_viewer(replay_export, out_path):
    with open(_TEMPLATE, "r", encoding="utf-8") as fh:
        template = fh.read()
    payload = json.dumps(replay_export, ensure_ascii=False)
    injected = template.replace(
        _MARKER, f"window.REPLAY = {payload};")
    if injected == template:
        raise RuntimeError("HTML 模板注入点缺失")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(injected)
    return out_path
