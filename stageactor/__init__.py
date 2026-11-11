"""虚拟舞台演员动作控制系统（确定性纯 Python 核心）。"""

from .math3d import *  # noqa: F401,F403
from .skeleton import Bone, Skeleton, Pose, fk, blend_pose, identity_pose
from .clip import Keyframe, EventMarker, AnimationClip
from .graph import Condition, Transition, State, StateGraph, Bundle
from .machine import ActorMachine
from .replay import ReplayRunner, FrameRecord

__all__ = [
    "Bone", "Skeleton", "Pose", "fk", "blend_pose", "identity_pose",
    "Keyframe", "EventMarker", "AnimationClip",
    "Condition", "Transition", "State", "StateGraph", "Bundle",
    "ActorMachine", "ReplayRunner", "FrameRecord",
]
