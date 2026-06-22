from lerobot.teleoperators.hand_teleop import HandTeleopConfig

__all__ = ["HandTeleop", "HandTeleopConfig"]


def __getattr__(name: str):
    if name == "HandTeleop":
        from lerobot.teleoperators.hand_teleop import HandTeleop

        return HandTeleop
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
