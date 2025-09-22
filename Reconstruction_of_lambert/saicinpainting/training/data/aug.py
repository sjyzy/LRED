# aug.py — Albumentations 1.x 兼容实现（无 imgaug 依赖）
# 保留旧类名，内部改为使用 A.Affine / A.Perspective

from typing import Optional, Tuple, Union
import albumentations as A
from albumentations.core.transforms_interface import DualTransform  # 供类型检查/兼容使用
import cv2
import logging

LOGGER = logging.getLogger(__name__)

# 兼容：项目里若从本文件 import DualIAATransform，则给个别名
DualIAATransform = DualTransform

Number = Union[int, float]
Range = Union[Tuple[Number, Number], Number]


def to_tuple(param, low=None):
    """仿 Albumentations 旧版的 to_tuple 行为：
       - None -> (low, low)
       - 标量 -> (param, param)
       - 序列 -> tuple(param)
    """
    if param is None:
        return (low, low)
    if isinstance(param, (list, tuple)):
        return tuple(param)
    return (param, param)


# ---------- 帮助函数：imgaug 风格参数到 OpenCV/Albumentations ----------
def _interp_from_order(order: int) -> int:
    """imgaug 的 order(0~4) -> OpenCV 插值枚举"""
    table = {
        0: cv2.INTER_NEAREST,
        1: cv2.INTER_LINEAR,
        2: cv2.INTER_CUBIC,
        3: cv2.INTER_AREA,
        4: cv2.INTER_LANCZOS4,
    }
    return table.get(int(order), cv2.INTER_LINEAR)


def _border_from_mode(mode: str) -> int:
    """imgaug 的边界模式 -> OpenCV"""
    mode = str(mode).lower()
    table = {
        "constant": cv2.BORDER_CONSTANT,
        "replicate": cv2.BORDER_REPLICATE,
        "reflect": cv2.BORDER_REFLECT_101,  # 更接近 imgaug 的 reflect 行为
        "reflect_101": cv2.BORDER_REFLECT_101,
        "wrap": cv2.BORDER_WRAP,
    }
    return table.get(mode, cv2.BORDER_REFLECT_101)


# ---------- 旧名：IAAAffine2（新实现：A.Affine） ----------
class IAAAffine2(A.Affine):
    """
    旧版基于 imgaug 的仿射增强替代实现，使用 Albumentations 的 A.Affine。
    尽量保持参数语义一致：
      - scale: 支持标量或区间，内部映射到 dict(x=..., y=...)
      - translate_percent / translate_px: 支持标量或区间；若均未提供，则默认 (0,0)（不平移）
      - rotate: 标量或区间（度）
      - shear: 标量或区间，内部映射为 dict(x=..., y=...)
      - order: imgaug 的插值 order -> OpenCV 插值
      - mode/cval: 边界与填充值映射
    """

    def __init__(
        self,
        scale: Range = (0.7, 1.3),
        translate_percent: Optional[Range] = None,
        translate_px: Optional[Range] = None,
        rotate: Range = 0.0,
        shear: Range = (-0.1, 0.1),
        order: int = 1,
        cval: Number = 0,
        mode: str = "reflect",
        always_apply: bool = False,
        p: float = 0.5,
        **kwargs,
    ):
        # 只传一个 translate 参数给 A.Affine，避免某些版本判断异常
        tp = None
        tpx = None
        if translate_px is not None:
            tpx = to_tuple(translate_px, 0)
        else:
            # 若没提供 px，则用 percent；若也未提供，则默认不平移 (0,0)
            tp = to_tuple(translate_percent if translate_percent is not None else (0, 0), 0)

        super().__init__(
            scale=dict(x=scale, y=scale),
            rotate=to_tuple(rotate),
            shear=dict(x=shear, y=shear),
            interpolation=_interp_from_order(order),
            cval=cval,
            mode=_border_from_mode(mode),
            fit_output=False,      # 与旧实现一致：保持输出尺寸
            keep_ratio=True,       # 缩放时保持比例，贴近旧行为
            translate_percent=tp,
            translate_px=tpx,
            always_apply=always_apply,
            p=p,
            **kwargs,
        )


# ---------- 旧名：IAAPerspective2（新实现：A.Perspective） ----------
class IAAPerspective2(A.Perspective):
    """
    旧版基于 imgaug 的透视增强替代实现，使用 Albumentations 的 A.Perspective。
    参数兼容：
      - scale: 与旧版一致（四点偏移强度）
      - keep_size: True 时保持输出尺寸
      - order/mode/cval: 用于插值和 pad 映射
    """

    def __init__(
        self,
        scale: Range = (0.05, 0.1),
        keep_size: bool = True,
        always_apply: bool = False,
        p: float = 0.5,
        order: int = 1,
        cval: Number = 0,
        mode: str = "replicate",
        **kwargs,
    ):
        super().__init__(
            scale=to_tuple(scale, 1.0),
            keep_size=keep_size,
            interpolation=_interp_from_order(order),
            pad_mode=_border_from_mode(mode),
            pad_val=cval,
            mask_pad_val=0,
            fit_output=not keep_size,   # 与 keep_size 对偶
            always_apply=always_apply,
            p=p,
            **kwargs,
        )


__all__ = [
    "DualTransform",
    "DualIAATransform",  # 兼容名
    "IAAAffine2",
    "IAAPerspective2",
]
