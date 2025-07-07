import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision                          # only needed for the optional MatteUNet idea
import numpy as np

from lib.net.BasePIFuNet import BasePIFuNet
from lib.net.FBNet       import GANLoss, IDMRFLoss, VGGLoss, define_D, define_G
from lib.net.net_util    import init_net


# --------------------------------------------------------------------------- #
# Helper: border-sampling foreground mask                                     #
# --------------------------------------------------------------------------- #
import torch

# --------------------------------------------------------------------------- #
# FIXED helper: border-sampling foreground mask                               #
# --------------------------------------------------------------------------- #
# Place this function inside apps/infer_s.py, before process_sample_combined
def make_foreground_mask_border(
        img: torch.Tensor,
        border_px: int = 20,
        tol: float = 0.01
    ) -> torch.Tensor:
    """
    Build a foreground mask by comparing each pixel to the average colour of the
    outer `border_px` pixels on all four sides.
    """
    assert img.ndim == 4 and img.size(1) == 3, "img must be (B,3,H,W)"
    B, C, H, W = img.shape
    border_px = min(border_px, H // 2, W // 2)
    k = border_px
    top, bottom = img[:, :, :k, :], img[:, :, H-k:, :]
    left, right = img[:, :, k:H-k, :k], img[:, :, k:H-k, W-k:]
    strips = [p.reshape(B, C, -1) for p in (top, bottom, left, right)]
    border_flat = torch.cat(strips, dim=2)
    bg_colour = border_flat.mean(dim=2, keepdim=True)
    dist = ((img - bg_colour.view(B, C, 1, 1))**2).sum(dim=1, keepdim=True).sqrt()
    mask = (dist > tol).float()
    return mask
# --------------------------------------------------------------------------- #
# Main network                                                                #
# --------------------------------------------------------------------------- #
class NormalNet_f(BasePIFuNet):
    """
    Dual-view normal-estimation network.

    * `netF` predicts front-view normals.
    * `netB` predicts back-view normals.
    * A border-sampling heuristic creates a foreground mask so that only the
      subject contributes to loss and output normals.

    All loss heads (L1, VGG, MRF, GAN) are configurable through `cfg.net`.
    """

    # --------------------------------------------------------------------- #
    # construction                                                          #
    # --------------------------------------------------------------------- #
    def __init__(self, cfg):
        super(NormalNet_f, self).__init__()

        self.opt = cfg.net

        # ----------------------- loss bookkeeping ------------------------ #
        self.F_losses       = [item[0] for item in self.opt.front_losses]
        self.B_losses       = [item[0] for item in self.opt.back_losses]
        self.F_losses_ratio = [item[1] for item in self.opt.front_losses]
        self.B_losses_ratio = [item[1] for item in self.opt.back_losses]
        self.ALL_losses     = self.F_losses + self.B_losses

        # ----------------------- loss functions -------------------------- #
        if self.training:
            if "vgg" in self.ALL_losses:
                self.vgg_loss = VGGLoss()
            if ("gan" in self.ALL_losses) or ("gan_feat" in self.ALL_losses):
                self.gan_loss = GANLoss(use_lsgan=True)
            if "mrf" in self.ALL_losses:
                self.mrf_loss = IDMRFLoss()
            if "l1" in self.ALL_losses:
                self.l1_loss = nn.SmoothL1Loss()

        # ----------------------- input channels -------------------------- #
        self.in_nmlF = [
            item[0] for item in self.opt.in_nml
            if "_F" in item[0] or item[0] == "image"
        ]
        self.in_nmlB = [
            item[0] for item in self.opt.in_nml
            if "_B" in item[0] or item[0] == "image_back"
        ]

        self.in_nmlF_dim = sum(
            item[1] for item in self.opt.in_nml
            if "_F" in item[0] or item[0] == "image"
        )
        self.in_nmlB_dim = sum(
            item[1] for item in self.opt.in_nml
            if "_B" in item[0] or item[0] == "image_back"
        )

        # ----------------------- generators & discriminator -------------- #
        self.netF = define_G(self.in_nmlF_dim, 3, 64, "global", 4, 9, 1, 3, "instance")
        self.netB = define_G(self.in_nmlB_dim, 3, 64, "global", 4, 9, 1, 3, "instance")

        if "gan" in self.ALL_losses:
            self.netD = define_D(
                3, 64, 3, "instance", False, 2, "gan_feat" in self.ALL_losses
            )

        init_net(self)

    # --------------------------------------------------------------------- #
    # forward pass                                                          #
    # --------------------------------------------------------------------- #
    def forward(self, in_tensor, out_dir=None):
        """
        Parameters
        ----------
        in_tensor : dict with keys 'image', 'image_back', plus any extra cues.
        out_dir   : (Optional) str – directory to write PNG visualisations.
        """
        # --------------- concatenate cues (front) ----------------------- #
        inF_list = [in_tensor[name] for name in self.in_nmlF]
        nmlF = self.netF(torch.cat(inF_list, dim=1))

        # --------------- concatenate cues (back) ------------------------ #
        inB_list = [
            in_tensor["image_back"] if name == "image_back" else in_tensor[name]
            for name in self.in_nmlB
        ]
        nmlB = self.netB(torch.cat(inB_list, dim=1))

        # --------------- unit-length normalisation ---------------------- #
        nmlF = nmlF / torch.norm(nmlF, dim=1, keepdim=True).clamp_min(1e-8)
        nmlB = nmlB / torch.norm(nmlB, dim=1, keepdim=True).clamp_min(1e-8)

        # --------------- save visuals (optional side-effect) ------------ #
        if out_dir is not None:
            mask      = make_foreground_mask_border(in_tensor["image"],
                                                    border_px=20, tol=0.10).detach()
            back_mask = make_foreground_mask_border(in_tensor["image_back"],
                                                    border_px=20, tol=0.10).detach()
            self.save_mask_image(mask, back_mask, out_dir)
            self.save_nml_image(nmlF, nmlB, out_dir)

        # --------------- Return raw network output ---------------------- #
        # The calling function is responsible for applying any masks.
        return nmlF, nmlB
        
    # --------------------------------------------------------------------- #
    # loss computation                                                      #
    # --------------------------------------------------------------------- #
    def get_norm_error(self, prd_F, prd_B, tgt):
        """
        Compute the multi-component loss between predicted and target normals.

        Returns
        -------
        dict mapping loss names to scalars.
        """
        tgt_F, tgt_B = tgt["normal_F"], tgt["normal_B"]
        total_loss = {"netF": 0.0, "netB": 0.0}

        # -------------------- L1 --------------------------------------- #
        if "l1" in self.F_losses:
            l1_F = self.l1_loss(prd_F, tgt_F)
            total_loss["netF"] += self.F_losses_ratio[self.F_losses.index("l1")] * l1_F
            total_loss["l1_F"] = self.F_losses_ratio[self.F_losses.index("l1")] * l1_F
        if "l1" in self.B_losses:
            l1_B = self.l1_loss(prd_B, tgt_B)
            total_loss["netB"] += self.B_losses_ratio[self.B_losses.index("l1")] * l1_B
            total_loss["l1_B"] = self.B_losses_ratio[self.B_losses.index("l1")] * l1_B

        # -------------------- VGG -------------------------------------- #
        if "vgg" in self.F_losses:
            vgg_F = self.vgg_loss(prd_F, tgt_F)
            total_loss["netF"] += self.F_losses_ratio[self.F_losses.index("vgg")] * vgg_F
            total_loss["vgg_F"] = self.F_losses_ratio[self.F_losses.index("vgg")] * vgg_F
        if "vgg" in self.B_losses:
            vgg_B = self.vgg_loss(prd_B, tgt_B)
            total_loss["netB"] += self.B_losses_ratio[self.B_losses.index("vgg")] * vgg_B
            total_loss["vgg_B"] = self.B_losses_ratio[self.B_losses.index("vgg")] * vgg_B

        # -------------------- MRF -------------------------------------- #
        scale_factor = 0.5
        if "mrf" in self.F_losses:
            mrf_F = self.mrf_loss(
                F.interpolate(prd_F, scale_factor=scale_factor, mode="bicubic",
                              align_corners=True),
                F.interpolate(tgt_F, scale_factor=scale_factor, mode="bicubic",
                              align_corners=True)
            )
            total_loss["netF"] += self.F_losses_ratio[self.F_losses.index("mrf")] * mrf_F
            total_loss["mrf_F"] = self.F_losses_ratio[self.F_losses.index("mrf")] * mrf_F
        if "mrf" in self.B_losses:
            mrf_B = self.mrf_loss(
                F.interpolate(prd_B, scale_factor=scale_factor, mode="bicubic",
                              align_corners=True),
                F.interpolate(tgt_B, scale_factor=scale_factor, mode="bicubic",
                              align_corners=True)
            )
            total_loss["netB"] += self.B_losses_ratio[self.B_losses.index("mrf")] * mrf_B
            total_loss["mrf_B"] = self.B_losses_ratio[self.B_losses.index("mrf")] * mrf_B

        # -------------------- GAN -------------------------------------- #
        if "gan" in self.ALL_losses:
            total_loss["netD"] = 0.0
            pred_fake = self.netD(prd_B)
            pred_real = self.netD(tgt_B)

            loss_D_fake = self.gan_loss(pred_fake, False)
            loss_D_real = self.gan_loss(pred_real, True)
            loss_G_fake = self.gan_loss(pred_fake, True)

            ratio_gan = self.B_losses_ratio[self.B_losses.index("gan")]
            total_loss["netD"] += 0.5 * (loss_D_fake + loss_D_real) * ratio_gan
            total_loss["D_fake"] = loss_D_fake * ratio_gan
            total_loss["D_real"] = loss_D_real * ratio_gan

            total_loss["netB"] += loss_G_fake * ratio_gan
            total_loss["G_fake"] = loss_G_fake * ratio_gan

            if "gan_feat" in self.ALL_losses:
                loss_G_feat = 0
                for i in range(2):
                    for j in range(len(pred_fake[i]) - 1):
                        loss_G_feat += self.gan_loss(pred_fake[i][j], True)

                ratio_feat = self.B_losses_ratio[self.B_losses.index("gan_feat")]
                total_loss["netB"] += loss_G_feat * ratio_feat
                total_loss["G_feat"] = loss_G_feat * ratio_feat

        return total_loss

    # --------------------------------------------------------------------- #
    # visualisation helpers                                                 #
    # --------------------------------------------------------------------- #
    def save_mask_image(self, mask, back_mask, out_dir):
        self.out_dir = out_dir

        mask      = (mask * 255).byte()
        back_mask = (back_mask * 255).byte()

        mask_img      = Image.fromarray(mask.squeeze().cpu().numpy(),      mode="L")
        back_mask_img = Image.fromarray(back_mask.squeeze().cpu().numpy(), mode="L")

        mask_img.save(f"{self.out_dir}/econ/png/mask_image.png")
        back_mask_img.save(f"{self.out_dir}/econ/png/mask_image_back.png")

    def save_nml_image(self, nmlF, nmlB, out_dir):
        self.out_dir = out_dir

        nmlF = ((nmlF + 1) / 2 * 255).byte()
        nmlB = ((nmlB + 1) / 2 * 255).byte()

        nmlF_img = Image.fromarray(
            nmlF.squeeze().permute(1, 2, 0).cpu().numpy(), mode="RGB"
        )
        nmlB_img = Image.fromarray(
            nmlB.squeeze().permute(1, 2, 0).cpu().numpy(), mode="RGB"
        )

        nmlF_img.save(f"{self.out_dir}/econ/png/nmlF_image.png")
        nmlB_img.save(f"{self.out_dir}/econ/png/nmlB_image.png")
