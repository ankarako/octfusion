from contextlib import nullcontext
from dataclasses import dataclass
from rich.console import Console
import os

import torch

k_TENSORBOARD_FOUND = False
try:
    from torch.utils.tensorboard import SummaryWriter
    k_TENSORBOARD_FOUND = True
except:
    k_TENSORBOARD_FOUND = False

from enum import Enum, auto


CONSOLE = Console(width=256)

@dataclass
class Palette:
    info: str = "#b2aa9b"
    warning: str = "#eca936"
    error: str = "#d06348"


class Spinners(Enum):
    Dots = auto()
    Dots2 = auto()
    Dots3 = auto()
    Dots4 = auto()
    Dots5 = auto()
    Dots6 = auto()
    Dots7 = auto()
    Dots8 = auto()
    Dots9 = auto()
    Dots10 = auto()
    Dots11 = auto()
    Dots12 = auto()
    Dots8Bit = auto()
    Line = auto()
    Line2 = auto()
    Pipe = auto()
    SimpleDots = auto()
    SimpleDotsScrolling = auto()
    Star = auto()
    Star2 = auto()
    Flip = auto()
    Hamburger = auto()
    GrowVertical = auto()
    GrowHorizontal = auto()
    Balloon = auto()
    Balloon2 = auto()
    Noise = auto()
    Bounce = auto()
    BoxBounce = auto()
    BoxBounce2 = auto()
    Triangle = auto()
    Arc = auto()
    Circle = auto()
    SquareCorners = auto()
    CircleQuarters = auto()
    CircleHalves = auto()
    Squish = auto()
    Toggle = auto()
    Toggle2 = auto()
    Toggle3 = auto()
    Toggle4 = auto()
    Toggle5 = auto()
    Toggle6 = auto()
    Toggle7 = auto()
    Toggle8 = auto()
    Toggle9 = auto()
    Toggle10 = auto()
    Toggle11 = auto()
    Toggle12 = auto()
    Toggle13 = auto()
    Arrow = auto()
    Arrow2 = auto()
    Arrow3 = auto()
    BouncingBar = auto()
    BouncingBall = auto()
    Smiley = auto()
    Monkey = auto()
    Hearts = auto()
    Clock = auto()
    Earth = auto()
    Material = auto()
    Moon = auto()
    Pong = auto()
    Shark = auto()
    Weather = auto()
    Grenade = auto()
    Christmas = auto()

k_spinner_to_str_table = {
    Spinners.Dots: "dots",
    Spinners.Dots2: "dots2",
    Spinners.Dots3: "dots3",
    Spinners.Dots4: "dots4",
    Spinners.Dots5: "dots5",
    Spinners.Dots6: "dots6",
    Spinners.Dots7: "dots7",
    Spinners.Dots8: "dots8",
    Spinners.Dots9: "dots9",
    Spinners.Dots10: "dots10",
    Spinners.Dots11: "dots11",
    Spinners.Dots12: "dots12",
    Spinners.Dots8Bit: "dots8Bit",
    Spinners.Line: "line",
    Spinners.Line2: "line2",
    Spinners.Pipe: "pipe",
    Spinners.SimpleDots: "simpleDots",
    Spinners.SimpleDotsScrolling: "simpleDotsScrolling",
    Spinners.Star: "star",
    Spinners.Star2: "star2",
    Spinners.Flip: "flip",
    Spinners.Hamburger: "hamburger",
    Spinners.GrowVertical: "growVertical",
    Spinners.GrowHorizontal: "growHorizontal",
    Spinners.Balloon: "balloon",
    Spinners.Balloon2: "balloon2",
    Spinners.Noise: "noise",
    Spinners.Bounce: "bounce",
    Spinners.BoxBounce: "boxBounce",
    Spinners.BoxBounce2: "boxBounce2",
    Spinners.Triangle: "triangle",
    Spinners.Arc: "arc",
    Spinners.Circle: "circle",
    Spinners.SquareCorners: "squareCorners",
    Spinners.CircleQuarters: "circleQuarters",
    Spinners.CircleHalves: "circleHalves",
    Spinners.Squish: "squish",
    Spinners.Toggle: "toggle",
    Spinners.Toggle2: "toggle2",
    Spinners.Toggle3: "toggle3",
    Spinners.Toggle4: "toggle4",
    Spinners.Toggle5: "toggle5",
    Spinners.Toggle6: "toggle6",
    Spinners.Toggle7: "toggle7",
    Spinners.Toggle8: "toggle8",
    Spinners.Toggle9: "toggle9",
    Spinners.Toggle10: "toggle10",
    Spinners.Toggle11: "toggle11",
    Spinners.Toggle12: "toggle12",
    Spinners.Toggle13: "toggle13",
    Spinners.Arrow: "arrow",
    Spinners.Arrow2: "arrow2",
    Spinners.Arrow3: "arrow3",
    Spinners.BouncingBar: "bouncingBar",
    Spinners.BouncingBall: "bouncingBall",
    Spinners.Smiley: "smiley",
    Spinners.Monkey: "monkey",
    Spinners.Hearts: "hearts",
    Spinners.Clock: "clock",
    Spinners.Earth: "earth",
    Spinners.Material: "material",
    Spinners.Moon: "moon",
    Spinners.Pong: "pong",
    Spinners.Shark: "shark",
    Spinners.Weather: "weather",
    Spinners.Grenade: "grenade",
    Spinners.Christmas: "christmas",
}




def INFO(msg: str) -> None:
    """
    Log an info message.

    :param msg The message to log.
    """
    CONSOLE.log(f"[{Palette.info}]{msg}[/{Palette.info}]")

def WARN(msg: str) -> None:
    """
    Log an info message.

    :param msg The message to log.
    """
    CONSOLE.log(f"[{Palette.warning}]{msg}[/{Palette.warning}]")

def ERROR(msg: str) -> None:
    """
    Log an info message.

    :param msg The message to log.
    """
    CONSOLE.log(f"[{Palette.error}]{msg}[/{Palette.error}]")


def status(msg: str, spinner: Spinners=Spinners.BouncingBall, verbose: bool=False):
    """
    Context manager that does nothing if verbose is ``True``.
    Otherwise it hides logs under ``msg``

    :param msg The message to log.
    :param spinner The spinner to use.
    :param verbose If ``True`` print all logs, otherwise hide them.
    """
    if verbose:
        return nullcontext()
    return CONSOLE.status(msg, spinner=k_spinner_to_str_table[spinner])


# Tensorboard instance
k_tboard_instance = None

def init_tensorboard(logdir: str, clear_history: bool=False) -> None:
    """
    Initialize the tensorboard logger.
    """
    global k_tboard_instance
    if not os.path.exists(logdir):
        ERROR(f"The specified log directory is invalid: {logdir}")
        return
    
    if clear_history:
        filenames = os.listdir(logdir)
        for filename in filenames:
            if 'events.out.tfevents' in filename:
                filepath = os.path.join(logdir, filename)
                os.remove(filepath)

    k_tboard_instance = SummaryWriter(log_dir=logdir)


def tboard_scalar(scalar: torch.Tensor, tag: str, step: int) -> None:
    """
    Log a scalar on tensorboard.

    :param scalar The scalar to log.
    :param name The of the logging window.
    :param step The current step.
    """
    global k_tboard_instance
    if k_TENSORBOARD_FOUND:
        
        k_tboard_instance.add_scalar(tag, scalar.detach(), step)


def tboard_image(image: torch.Tensor, tag: str, step: int, format: str="HWC") -> None:
    """
    Log an image on tensorboard
https://github.com/CompVis/latent-diffusion.githttps://github.com/CompVis/latent-diffusion.git
    :param image The image to log.
    :param tag 
    :param step
    """
    global k_tboard_instance
    if k_TENSORBOARD_FOUND:
        k_tboard_instance.add_image(tag, image, step, dataformats=format)