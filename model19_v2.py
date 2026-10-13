"""
model19_v2.py -- RTFNet-inspired ResNet-18 for RGB-Thermal semantic segmentation (V2).

Standalone: depends only on PyTorch + torchvision (never imports the teammate's model.py
and never imports model19.py). This module has no dataset-path assumptions; dataset19_v2.py
supplies RGB/thermal tensors from either the raw Kaggle RGBA layout or the prepared layout.

What this is (and is not)
-------------------------
* Reference: Sun et al., "RTFNet: RGB-Thermal Fusion Network for Semantic Segmentation of
  Urban Scenes", IEEE RA-L 2019, and the official code (github.com/yuxiangsun/RTFNet,
  model/RTFNet.py).  This is an "RTFNet-inspired ResNet-18".  It does NOT reproduce any
  published RTFNet benchmark and must not be reported as doing so.

Architecture (verified against the official forward pass; unchanged from V1)
---------------------------------------------------------------------------
Two separate ResNet-18 encoders (RGB: 3 ch, thermal: 1 ch).  The thermal features are ADDED
into the RGB stream (element-wise) after the stem, layer1, layer2, layer3 and layer4; the
thermal stream itself is never modified by the fusion.  The decoder ("Upception") has five
stages (plain TransBottleneck + 2x up-sampling TransBottleneck): 512 -> 256 -> 128 -> 64 -> 32.

V1 -> V2 changes (each is a concrete, documented decision)
----------------------------------------------------------
1. Classifier head (`head=`).
   * "official": the last decoder stage maps 32 -> num_classes with BatchNorm + residual on the
     class scores themselves (as in the official code).  With ``final_relu=False`` (V1 default)
     the output is raw logits.  BatchNorm applied directly to the class channels re-normalises
     every class score per batch, which fights the very skewed class priors of MFNet
     (class 0 = 92 % of pixels) and makes train-mode (batch statistics) and eval-mode
     (running statistics) logits differ most exactly at the layer that decides the argmax.
   * "conv1x1" (V2 DEFAULT): the last stage keeps 32 channels at full resolution and a plain
     1x1 convolution (with bias) produces the logits - no BN/ReLU on the class scores.
     This is a hypothesis-driven change, NOT a proven fix: compare both with ``--head``.
2. Optional class-prior bias initialisation (`prior_probs=`): the classifier bias can be
   initialised to log(prior) so the first epochs do not have to learn the 92 % background
   prior from scratch.  Off by default.  Only valid for head="conv1x1".
3. Optional frozen encoder BatchNorm statistics (`freeze_encoder_bn`, see
   ``set_train_mode``): useful when the batch size is 1-2 (noisy BN statistics).  Off by default.
4. Thermal conv1 initialisation: ``thermal_init="mean"`` (official: mean of the RGB filters
   over the input-channel axis; default) or ``"sum"`` (response equals feeding a grey image
   replicated to 3 channels to the pretrained net).
5. Pretrained weights can come from a local file (``weights_path=``) so no download is needed.
   If weights cannot be obtained the model builds with random weights and
   ``model.pretrained_info`` says so exactly; ``--require-pretrained`` in the training
   script aborts instead.
6. Inputs not divisible by 32 are edge-padded internally and the logits cropped back.
7. ``forward`` never substitutes zeros/synthetic data for a missing thermal tensor: wrong
   shapes/devices raise ``ValueError``.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

NUM_CLASSES = 9
ENCODER_STRIDE = 32
HEADS = ("conv1x1", "official")
_RESNET18_IMAGENET_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"


# --------------------------------------------------------------------------- #
# Pretrained-weight helper
# --------------------------------------------------------------------------- #
def load_imagenet_resnet18_state_dict(weights_path: Optional[str] = None) -> Tuple[Optional[dict], str]:
    """Obtain ImageNet ResNet-18 weights.  Never raises.  Returns (state_dict_or_None, message)."""
    errors = []
    if weights_path:
        try:
            state = torch.load(weights_path, map_location="cpu")
            if "conv1.weight" not in state:
                raise KeyError("file does not look like a torchvision resnet18 state_dict")
            return state, f"local file {weights_path}"
        except Exception as exc:  # noqa: BLE001
            errors.append(f"local file: {type(exc).__name__}: {exc}")
    try:
        weights = tv_models.ResNet18_Weights.IMAGENET1K_V1
        return weights.get_state_dict(progress=True), "torchvision ResNet18_Weights.IMAGENET1K_V1"
    except Exception as exc:  # noqa: BLE001
        errors.append(f"torchvision weights API: {type(exc).__name__}: {exc}")
    try:
        state = torch.hub.load_state_dict_from_url(_RESNET18_IMAGENET_URL, progress=True)
        return state, "resnet18-f37072fd.pth (direct URL)"
    except Exception as exc:  # noqa: BLE001
        errors.append(f"direct URL: {type(exc).__name__}: {exc}")
    return None, " | ".join(errors)


# --------------------------------------------------------------------------- #
# Decoder building blocks (Upception)
# --------------------------------------------------------------------------- #
def _init_decoder_module(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_uniform_(module.weight)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class TransBottleneck(nn.Module):
    """Official RTFNet 'TransBottleneck' (1x1 -> 3x3 -> 3x3 / 2x2-transpose + residual)."""

    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                 upsample: Optional[nn.Module] = None, apply_relu: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if upsample is not None and stride != 1:
            self.conv3 = nn.ConvTranspose2d(planes, planes, kernel_size=2, stride=stride,
                                            padding=0, bias=False)
        else:
            self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride
        self.apply_relu = apply_relu
        for m in self.modules():
            _init_decoder_module(m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.upsample is not None:
            residual = self.upsample(x)
        out = out + residual
        return self.relu(out) if self.apply_relu else out


def make_upception_stage(inplanes: int, planes: int, final_block_relu: bool = True) -> nn.Sequential:
    """[TransBottleneck(in,in)] + [TransBottleneck(in->planes, x2 up-sampling)]."""
    shortcut = nn.Sequential(
        nn.ConvTranspose2d(inplanes, planes, kernel_size=2, stride=2, padding=0, bias=False),
        nn.BatchNorm2d(planes),
    )
    for m in shortcut.modules():
        _init_decoder_module(m)
    return nn.Sequential(
        TransBottleneck(inplanes, inplanes),
        TransBottleneck(inplanes, planes, stride=2, upsample=shortcut, apply_relu=final_block_relu),
    )


# --------------------------------------------------------------------------- #
# The network
# --------------------------------------------------------------------------- #
class RTFNet18V2(nn.Module):
    """RTFNet-inspired dual-encoder (ResNet-18) RGB-T segmentation network (V2)."""

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True,
                 head: str = "conv1x1", final_relu: bool = False,
                 thermal_init: str = "mean", weights_path: Optional[str] = None,
                 imagenet_state_dict: Optional[dict] = None,
                 prior_probs: Optional[Sequence[float]] = None,
                 freeze_encoder_bn: bool = False):
        super().__init__()
        if not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError(f"num_classes must be an int >= 2, got {num_classes!r}")
        if head not in HEADS:
            raise ValueError(f"head must be one of {HEADS}, got {head!r}")
        if thermal_init not in ("mean", "sum"):
            raise ValueError("thermal_init must be 'mean' or 'sum'")
        if prior_probs is not None:
            if head != "conv1x1":
                raise ValueError("prior_probs bias initialisation needs head='conv1x1'")
            if len(prior_probs) != num_classes:
                raise ValueError(f"prior_probs needs {num_classes} entries, got {len(prior_probs)}")
        self.num_classes, self.head_type, self.final_relu = num_classes, head, final_relu
        self.freeze_encoder_bn = bool(freeze_encoder_bn)

        # ---- pretrained weights (optional, never fatal) -------------------- #
        self.pretrained_info: Dict[str, object] = {
            "requested": bool(pretrained), "loaded": False, "source": None, "message": "",
            "head": head, "thermal_init": thermal_init}
        state = None
        if pretrained:
            if imagenet_state_dict is not None:
                state, msg = imagenet_state_dict, "user-supplied state_dict"
            else:
                state, msg = load_imagenet_resnet18_state_dict(weights_path)
            if state is None:
                self.pretrained_info["message"] = "ImageNet weights NOT loaded (random init used). " + msg
                print("[model19_v2] WARNING: " + self.pretrained_info["message"], file=sys.stderr)
            else:
                self.pretrained_info.update(loaded=True, source=msg,
                                            message=f"ImageNet weights loaded from {msg}")
        else:
            self.pretrained_info["message"] = "pretrained=False -> random initialisation"

        rgb_resnet = tv_models.resnet18(weights=None)
        thermal_resnet = tv_models.resnet18(weights=None)
        if state is not None:
            rgb_resnet.load_state_dict(state)
            thermal_resnet.load_state_dict(state)

        # ---- RGB ENCODER ---------------------------------------------------- #
        self.encoder_rgb_conv1 = rgb_resnet.conv1
        self.encoder_rgb_bn1 = rgb_resnet.bn1
        self.encoder_rgb_relu = rgb_resnet.relu
        self.encoder_rgb_maxpool = rgb_resnet.maxpool
        self.encoder_rgb_layer1 = rgb_resnet.layer1
        self.encoder_rgb_layer2 = rgb_resnet.layer2
        self.encoder_rgb_layer3 = rgb_resnet.layer3
        self.encoder_rgb_layer4 = rgb_resnet.layer4

        # ---- THERMAL ENCODER ------------------------------------------------ #
        self.encoder_thermal_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if state is not None:
            with torch.no_grad():
                w = state["conv1.weight"]
                w1 = w.mean(dim=1, keepdim=True) if thermal_init == "mean" else w.sum(dim=1, keepdim=True)
                self.encoder_thermal_conv1.weight.copy_(w1)
        else:
            nn.init.kaiming_normal_(self.encoder_thermal_conv1.weight, mode="fan_out", nonlinearity="relu")
        self.encoder_thermal_bn1 = thermal_resnet.bn1
        self.encoder_thermal_relu = thermal_resnet.relu
        self.encoder_thermal_maxpool = thermal_resnet.maxpool
        self.encoder_thermal_layer1 = thermal_resnet.layer1
        self.encoder_thermal_layer2 = thermal_resnet.layer2
        self.encoder_thermal_layer3 = thermal_resnet.layer3
        self.encoder_thermal_layer4 = thermal_resnet.layer4

        # ---- DECODER --------------------------------------------------------- #
        self.deconv1 = make_upception_stage(512, 256)
        self.deconv2 = make_upception_stage(256, 128)
        self.deconv3 = make_upception_stage(128, 64)
        self.deconv4 = make_upception_stage(64, 32)
        if head == "official":
            self.deconv5 = make_upception_stage(32, num_classes, final_block_relu=final_relu)
            self.classifier = None
        else:
            self.deconv5 = make_upception_stage(32, 32, final_block_relu=True)
            self.classifier = nn.Conv2d(32, num_classes, kernel_size=1, bias=True)
            nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.classifier.bias)
            if prior_probs is not None:
                p = torch.as_tensor(list(prior_probs), dtype=torch.float32).clamp_min(1e-6)
                p = p / p.sum()
                with torch.no_grad():
                    self.classifier.bias.copy_(p.log())
                self.pretrained_info["prior_bias_init"] = True

    # ------------------------------------------------------------------ #
    def encoder_bn_modules(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.BatchNorm2d) and (name.startswith("encoder_rgb") or
                                                   name.startswith("encoder_thermal")):
                yield m

    def set_train_mode(self) -> "RTFNet18V2":
        """model.train(), but keep encoder BatchNorm layers in eval mode if freeze_encoder_bn."""
        self.train()
        if self.freeze_encoder_bn:
            for m in self.encoder_bn_modules():
                m.eval()
        return self

    @staticmethod
    def _check_inputs(rgb: torch.Tensor, thermal: torch.Tensor) -> None:
        if not (torch.is_tensor(rgb) and torch.is_tensor(thermal)):
            raise TypeError("rgb and thermal must both be torch.Tensors")
        if rgb.dim() != 4 or rgb.shape[1] != 3:
            raise ValueError(f"rgb must have shape [B, 3, H, W], got {tuple(rgb.shape)}")
        if thermal.dim() != 4 or thermal.shape[1] != 1:
            raise ValueError(f"thermal must have shape [B, 1, H, W], got {tuple(thermal.shape)}")
        if rgb.shape[0] != thermal.shape[0] or rgb.shape[2:] != thermal.shape[2:]:
            raise ValueError("rgb and thermal must share batch size and spatial size: "
                             f"rgb {tuple(rgb.shape)} vs thermal {tuple(thermal.shape)}")
        if min(rgb.shape[2:]) < ENCODER_STRIDE:
            raise ValueError(f"H and W must be >= {ENCODER_STRIDE}, got {tuple(rgb.shape[2:])}")
        if rgb.device != thermal.device:
            raise ValueError(f"rgb ({rgb.device}) and thermal ({thermal.device}) are on different devices")

    def forward(self, rgb: torch.Tensor, thermal: torch.Tensor) -> torch.Tensor:
        self._check_inputs(rgb, thermal)
        h, w = rgb.shape[-2:]
        pad_h, pad_w = (-h) % ENCODER_STRIDE, (-w) % ENCODER_STRIDE
        if pad_h or pad_w:
            rgb = F.pad(rgb, (0, pad_w, 0, pad_h), mode="replicate")
            thermal = F.pad(thermal, (0, pad_w, 0, pad_h), mode="replicate")

        r = self.encoder_rgb_relu(self.encoder_rgb_bn1(self.encoder_rgb_conv1(rgb)))             # 64, H/2
        t = self.encoder_thermal_relu(self.encoder_thermal_bn1(self.encoder_thermal_conv1(thermal)))
        r = r + t                                                                                # fusion 1
        r = self.encoder_rgb_maxpool(r)
        t = self.encoder_thermal_maxpool(t)

        r = self.encoder_rgb_layer1(r); t = self.encoder_thermal_layer1(t); r = r + t            # fusion 2
        r = self.encoder_rgb_layer2(r); t = self.encoder_thermal_layer2(t); r = r + t            # fusion 3
        r = self.encoder_rgb_layer3(r); t = self.encoder_thermal_layer3(t); r = r + t            # fusion 4
        r = self.encoder_rgb_layer4(r); t = self.encoder_thermal_layer4(t)
        fuse = r + t                                                                             # fusion 5

        x = self.deconv1(fuse)     # 256, H/16
        x = self.deconv2(x)        # 128, H/8
        x = self.deconv3(x)        # 64,  H/4
        x = self.deconv4(x)        # 32,  H/2
        x = self.deconv5(x)        # head=official: C at H | head=conv1x1: 32 at H
        logits = self.classifier(x) if self.classifier is not None else x
        if pad_h or pad_w:
            logits = logits[..., :h, :w]
        return logits              # raw logits [B, C, H, W]; no softmax before CrossEntropyLoss


# --------------------------------------------------------------------------- #
# Public helpers
# --------------------------------------------------------------------------- #
def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True, head: str = "conv1x1",
                final_relu: bool = False, thermal_init: str = "mean",
                weights_path: Optional[str] = None, imagenet_state_dict: Optional[dict] = None,
                prior_probs: Optional[Sequence[float]] = None, freeze_encoder_bn: bool = False,
                device: Optional[torch.device | str] = None) -> RTFNet18V2:
    model = RTFNet18V2(num_classes=num_classes, pretrained=pretrained, head=head,
                       final_relu=final_relu, thermal_init=thermal_init, weights_path=weights_path,
                       imagenet_state_dict=imagenet_state_dict, prior_probs=prior_probs,
                       freeze_encoder_bn=freeze_encoder_bn)
    return model.to(device) if device is not None else model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    return total, sum(p.numel() for p in model.parameters() if p.requires_grad)


def parameter_group(name: str) -> str:
    if name.startswith("encoder_rgb"):
        return "rgb_encoder"
    if name.startswith("encoder_thermal"):
        return "thermal_encoder"
    return "decoder"


def parameter_breakdown(model: nn.Module) -> Dict[str, int]:
    groups = {"rgb_encoder": 0, "thermal_encoder": 0, "decoder": 0}
    for name, p in model.named_parameters():
        groups[parameter_group(name)] += p.numel()
    return groups


def describe_model(model: RTFNet18V2) -> str:
    total, trainable = count_parameters(model)
    parts = parameter_breakdown(model)
    return (f"RTFNet18V2 head={model.head_type} | params {total:,} total ({trainable:,} trainable) | "
            f"RGB enc {parts['rgb_encoder']:,} | thermal enc {parts['thermal_encoder']:,} | "
            f"decoder+head {parts['decoder']:,} | {model.pretrained_info['message']}")


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def loss_backward_test(model: RTFNet18V2, rgb: torch.Tensor, thermal: torch.Tensor,
                       target: torch.Tensor, criterion=None) -> Dict[str, object]:
    """Real-loss / backward test interface.

    Feed a REAL batch (rgb [B,3,H,W], thermal [B,1,H,W], target [B,H,W] int64).  Runs forward,
    a loss (``criterion(logits, target) -> loss`` or ``(loss, parts)``; default cross-entropy
    with ignore_index=255), backward, and checks that every parameter in the RGB encoder,
    thermal encoder and decoder received a finite gradient and that each group has non-zero
    gradients.  Raises AssertionError on failure; returns a summary dict.
    """
    model.set_train_mode()
    model.zero_grad(set_to_none=True)
    logits = model(rgb, thermal)
    assert logits.shape == (rgb.shape[0], model.num_classes, *rgb.shape[2:]), tuple(logits.shape)
    if criterion is None:
        loss = F.cross_entropy(logits.float(), target, ignore_index=255)
    else:
        out = criterion(logits.float(), target)
        loss = out[0] if isinstance(out, tuple) else out
    assert torch.isfinite(loss), f"non-finite loss {loss}"
    loss.backward()
    groups = {"rgb_encoder": [0, 0], "thermal_encoder": [0, 0], "decoder": [0, 0]}
    for n, p in model.named_parameters():
        assert p.grad is not None, f"no gradient for {n}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient in {n}"
        g = groups[parameter_group(n)]
        g[0] += 1
        g[1] += int(p.grad.abs().sum() > 0)
    for k, (n_all, n_nz) in groups.items():
        assert n_nz > 0, f"all-zero gradients in {k}"
    model.zero_grad(set_to_none=True)
    return {"loss": float(loss.detach()), "grad_tensors_nonzero": {k: f"{v[1]}/{v[0]}" for k, v in groups.items()}}


def run_unit_test(device: str = "cpu", pretrained: bool = False, height: int = 480, width: int = 640,
                  batch: int = 1, head: str = "conv1x1", weights_path: Optional[str] = None) -> None:
    """Forward-pass test on RANDOM tensors (shape/finite checks only - explicitly synthetic)."""
    dev = torch.device(device)
    model = build_model(NUM_CLASSES, pretrained=pretrained, head=head, weights_path=weights_path,
                        device=dev).eval()
    print(describe_model(model))
    rgb = torch.randn(batch, 3, height, width, device=dev)
    thermal = torch.randn(batch, 1, height, width, device=dev)
    print(f"[synthetic random inputs, shape test only] RGB {list(rgb.shape)}  Thermal {list(thermal.shape)}")
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(rgb, thermal)
    print(f"Output logits {list(out.shape)} ({time.perf_counter() - t0:.2f}s, device={dev})")
    assert tuple(out.shape) == (batch, NUM_CLASSES, height, width)
    assert torch.isfinite(out).all() and out.dtype == torch.float32
    with torch.no_grad():
        odd = model(torch.randn(1, 3, 100, 150, device=dev), torch.randn(1, 1, 100, 150, device=dev))
    assert tuple(odd.shape) == (1, NUM_CLASSES, 100, 150), tuple(odd.shape)
    print(f"Odd-size check: [1,3,100,150] -> {list(odd.shape)} OK")
    for bad_rgb, bad_th, label in [
        (torch.randn(1, 4, 64, 64, device=dev), torch.randn(1, 1, 64, 64, device=dev), "4-channel rgb"),
        (torch.randn(1, 3, 64, 64, device=dev), torch.randn(1, 3, 64, 64, device=dev), "3-channel thermal"),
        (torch.randn(1, 3, 64, 64, device=dev), torch.randn(1, 1, 64, 96, device=dev), "size mismatch"),
    ]:
        try:
            model(bad_rgb, bad_th)
        except ValueError:
            print(f"Bad-input check: {label}: rejected OK")
        else:
            raise AssertionError(f"model accepted invalid input: {label}")
    # thermal must actually influence the output (no silent bypass)
    with torch.no_grad():
        a = model(rgb[:, :, :64, :64], thermal[:, :, :64, :64])
        b = model(rgb[:, :, :64, :64], thermal[:, :, :64, :64] + 3.0)
    assert (a - b).abs().max() > 0, "thermal input has no effect on the output"
    print("Thermal-influence check: output changes when thermal changes OK")
    print("MODEL19_V2 UNIT TEST PASSED")


def run_real_batch_test(data_root: Optional[str], device: str, pretrained: bool, head: str,
                        img_size: Tuple[int, int], batch: int, weights_path: Optional[str]) -> None:
    """Real-data loss/backward test: needs dataset19_v2.py and the MFNet_CMX folder."""
    import dataset19_v2 as d2   # local import: the model file itself stays dataset-independent
    root = d2.resolve_data_root(data_root)
    ds = d2.RGBTDataset(root, "train", img_size=img_size, augment=False)
    rgbs, ths, ys = zip(*[ds[i] for i in range(batch)])
    dev = torch.device(device)
    rgb, th, y = (torch.stack(rgbs).to(dev), torch.stack(ths).to(dev), torch.stack(ys).to(dev))
    model = build_model(NUM_CLASSES, pretrained=pretrained, head=head, weights_path=weights_path, device=dev)
    print(describe_model(model))
    res = loss_backward_test(model, rgb, th, y)
    print(f"Real batch ({batch} samples from {root}): loss {res['loss']:.4f}, "
          f"non-zero grad tensors {res['grad_tensors_nonzero']}")
    print("MODEL19_V2 REAL-BATCH LOSS/BACKWARD TEST PASSED")


def main() -> None:
    p = argparse.ArgumentParser(description="RTFNet18V2 tests")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--pretrained", action="store_true", help="load ImageNet weights (download on first use)")
    p.add_argument("--weights-path", default=None, help="local resnet18 state_dict (.pth)")
    p.add_argument("--head", choices=HEADS, default="conv1x1")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--real-batch", action="store_true",
                   help="run the loss/backward test on real MFNet samples (needs the dataset)")
    p.add_argument("--data-root", default=None)
    a = p.parse_args()
    if a.real_batch:
        run_real_batch_test(a.data_root, a.device, a.pretrained, a.head, (a.height, a.width),
                            a.batch, a.weights_path)
    else:
        run_unit_test(a.device, a.pretrained, a.height, a.width, a.batch, a.head, a.weights_path)


if __name__ == "__main__":
    main()
