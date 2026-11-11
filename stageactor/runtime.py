"""部署包运行器：无编辑器环境下加载发布包、校验资源摘要并真实回放自检。

启动流程刻意 fail-closed：先核对清单与资源引用，再执行旧样例；自检未全部
通过时不保留证据文件，调用方必须拒绝采用该次产物。
"""

import os

from .release import ReleaseError, load_manifest, load_package, run_selfcheck


def _discard(path):
    if os.path.exists(path):
        os.remove(path)


def start_package(package_dir, out_dir=None):
    """启动发布包并执行真实任务（旧样例回放），返回自检证据。

    先清掉上一次的证据再加载资源：清单损坏或资源缺失时同样不会留下可被
    误认为“本次通过”的旧产物。
    """
    package_dir = os.path.abspath(package_dir)
    out_dir = os.path.abspath(out_dir) if out_dir else package_dir
    os.makedirs(out_dir, exist_ok=True)
    _discard(os.path.join(out_dir, "evidence.json"))
    try:
        manifest = load_manifest(os.path.join(package_dir, "manifest.json"))
    except ReleaseError:
        raise
    _discard(os.path.join(out_dir, manifest["evidence"]))
    manifest, bundle, config = load_package(package_dir)
    return run_selfcheck(bundle, config, out_dir, manifest=manifest)

