import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.net.BasePIFuNet import BasePIFuNet
from lib.net.FBNet import GANLoss, IDMRFLoss, VGGLoss, define_D, define_G
from lib.net.net_util import init_net
from PIL import Image

class NormalNet_f(BasePIFuNet):
    """
    HG PIFu network uses Hourglass stacks as the image filter.
    It does the following:
        1. Compute image feature stacks and store it in self.im_feat_list
            self.im_feat_list[-1] is the last stack (output stack)
        2. Calculate calibration
        3. If training, it index on every intermediate stacks,
            If testing, it index on the last stack.
        4. Classification.
        5. During training, error is calculated on all stacks.
    """
    def __init__(self, cfg):
        super(NormalNet_f, self).__init__()

        self.opt = cfg.net

        # Loss functions
        self.F_losses = [item[0] for item in self.opt.front_losses]
        self.B_losses = [item[0] for item in self.opt.back_losses]
        self.F_losses_ratio = [item[1] for item in self.opt.front_losses]
        self.B_losses_ratio = [item[1] for item in self.opt.back_losses]
        self.ALL_losses = self.F_losses + self.B_losses

        if self.training:
            if 'vgg' in self.ALL_losses:
                self.vgg_loss = VGGLoss()
            if ('gan' in self.ALL_losses) or ('gan_feat' in self.ALL_losses):
                self.gan_loss = GANLoss(use_lsgan=True)
            if 'mrf' in self.ALL_losses:
                self.mrf_loss = IDMRFLoss()
            if 'l1' in self.ALL_losses:
                self.l1_loss = nn.SmoothL1Loss()

         #Define the input names based on updated configuration
        self.in_nmlF = [
            item[0] for item in self.opt.in_nml if "_F" in item[0]  
            or item[0] == "image"  
        ]
        self.in_nmlB = [
            item[0] for item in self.opt.in_nml if "_B" in item[0] 
            or item[0] == "image_back"  
        ]

        # Calculate input dimensions for front and back
        self.in_nmlF_dim = sum([
            item[1] for item in self.opt.in_nml if "_F" in item[0]  
            or item[0] == "image"
        ])
        self.in_nmlB_dim = sum([
            item[1] for item in self.opt.in_nml if "_B" in item[0]  
            or item[0] == "image_back"  
        ])

        # Define the networks for front and back normal estimation
        self.netF = define_G(self.in_nmlF_dim, 3, 64, "global", 4, 9, 1, 3, "instance")
        self.netB = define_G(self.in_nmlB_dim, 3, 64, "global", 4, 9, 1, 3, "instance")

        # Define the discriminator if GAN loss is used
        if 'gan' in self.ALL_losses:
            self.netD = define_D(3, 64, 3, 'instance', False, 2, 'gan_feat' in self.ALL_losses)

        init_net(self)

    def forward(self, in_tensor):
        # Prepare input lists for forward (front) and backward (back) normal estimation
        inF_list = []
        inB_list = []

        # Use front image for calculating front normals
        for name in self.in_nmlF:
            inF_list.append(in_tensor[name])  # `image`, `T_normal_F`, etc.

        # Use back image for calculating back normals
        for name in self.in_nmlB:
            if name == "image_back":
                inB_list.append(in_tensor["image_back"])  
            else:
                inB_list.append(in_tensor[name])  # For shared inputs 

        # Concatenate inputs along the channel dimension
        nmlF = self.netF(torch.cat(inF_list, dim=1))  # Forward network
        nmlB = self.netB(torch.cat(inB_list, dim=1))  # Backward network

        # Normalize normals to ensure ||normal|| == 1
        nmlF_normalized = nmlF / torch.norm(nmlF, dim=1, keepdim=True)
        nmlB_normalized = nmlB / torch.norm(nmlB, dim=1, keepdim=True)

        # Create a mask for valid pixels (non-zero regions)
        mask = ((in_tensor["image"].abs().sum(dim=1, keepdim=True) != 0.0).detach().float())
        back_mask = ((in_tensor["image_back"].abs().sum(dim=1, keepdim=True) != 0.0).detach().float())

        self.save_mask_image(mask, back_mask)
        self.save_nml_image(nmlF_normalized, nmlB_normalized)
        # Return masked and normalized normals
        return nmlF_normalized * mask, nmlB_normalized * back_mask

    def get_norm_error(self, prd_F, prd_B, tgt):
        """Calculate normal loss

        Args:
            prd_F (torch.tensor): Predicted front normals [B, 3, H, W]
            prd_B (torch.tensor): Predicted back normals [B, 3, H, W]
            tgt (dict): Target normals, contains "normal_F" and "normal_B"
        """
        tgt_F, tgt_B = tgt["normal_F"], tgt["normal_B"]

        # Initialize loss dictionary
        total_loss = {"netF": 0.0, "netB": 0.0}

        # Front loss
        if 'l1' in self.F_losses:
            l1_F_loss = self.l1_loss(prd_F, tgt_F)
            total_loss["netF"] += self.F_losses_ratio[self.F_losses.index('l1')] * l1_F_loss
            total_loss["l1_F"] = self.F_losses_ratio[self.F_losses.index('l1')] * l1_F_loss
        
        # Back loss
        if 'l1' in self.B_losses:
            l1_B_loss = self.l1_loss(prd_B, tgt_B)
            total_loss["netB"] += self.B_losses_ratio[self.B_losses.index('l1')] * l1_B_loss
            total_loss["l1_B"] = self.B_losses_ratio[self.B_losses.index('l1')] * l1_B_loss

        # VGG loss for front
        if 'vgg' in self.F_losses:
            vgg_F_loss = self.vgg_loss(prd_F, tgt_F)
            total_loss["netF"] += self.F_losses_ratio[self.F_losses.index('vgg')] * vgg_F_loss
            total_loss["vgg_F"] = self.F_losses_ratio[self.F_losses.index('vgg')] * vgg_F_loss
        
        # VGG loss for back
        if 'vgg' in self.B_losses:
            vgg_B_loss = self.vgg_loss(prd_B, tgt_B)
            total_loss["netB"] += self.B_losses_ratio[self.B_losses.index('vgg')] * vgg_B_loss
            total_loss["vgg_B"] = self.B_losses_ratio[self.B_losses.index('vgg')] * vgg_B_loss

        # Multi-resolution feature loss for front
        scale_factor = 0.5
        if 'mrf' in self.F_losses:
            mrf_F_loss = self.mrf_loss(
                F.interpolate(prd_F, scale_factor=scale_factor, mode='bicubic', align_corners=True),
                F.interpolate(tgt_F, scale_factor=scale_factor, mode='bicubic', align_corners=True)
            )
            total_loss["netF"] += self.F_losses_ratio[self.F_losses.index('mrf')] * mrf_F_loss
            total_loss["mrf_F"] = self.F_losses_ratio[self.F_losses.index('mrf')] * mrf_F_loss
        
        # Multi-resolution feature loss for back
        if 'mrf' in self.B_losses:
            mrf_B_loss = self.mrf_loss(
                F.interpolate(prd_B, scale_factor=scale_factor, mode='bicubic', align_corners=True),
                F.interpolate(tgt_B, scale_factor=scale_factor, mode='bicubic', align_corners=True)
            )
            total_loss["netB"] += self.B_losses_ratio[self.B_losses.index('mrf')] * mrf_B_loss
            total_loss["mrf_B"] = self.B_losses_ratio[self.B_losses.index('mrf')] * mrf_B_loss

        # GAN loss
        if 'gan' in self.ALL_losses:
            total_loss["netD"] = 0.0
            pred_fake = self.netD.forward(prd_B)
            pred_real = self.netD.forward(tgt_B)
            loss_D_fake = self.gan_loss(pred_fake, False)
            loss_D_real = self.gan_loss(pred_real, True)
            loss_G_fake = self.gan_loss(pred_fake, True)

            total_loss["netD"] += 0.5 * (loss_D_fake + loss_D_real) * self.B_losses_ratio[self.B_losses.index('gan')]
            total_loss["D_fake"] = loss_D_fake * self.B_losses_ratio[self.B_losses.index('gan')]
            total_loss["D_real"] = loss_D_real * self.B_losses_ratio[self.B_losses.index('gan')]

            total_loss["netB"] += loss_G_fake * self.B_losses_ratio[self.B_losses.index('gan')]
            total_loss["G_fake"] = loss_G_fake * self.B_losses_ratio[self.B_losses.index('gan')]

            if 'gan_feat' in self.ALL_losses:
                loss_G_GAN_Feat = 0
                for i in range(2):
                    for j in range(len(pred_fake[i]) - 1):
                        loss_G_GAN_Feat += self.gan_loss(pred_fake[i][j], True)

                total_loss["netB"] += loss_G_GAN_Feat * self.B_losses_ratio[self.B_losses.index('gan_feat')]
                total_loss["G_feat"] = loss_G_GAN_Feat * self.B_losses_ratio[self.B_losses.index('gan_feat')]

        return total_loss

    def save_mask_image(self, mask, back_mask):
        # Normalize the tensors to range 0-255 (optional if they're already 0 or 1)
        mask = (mask * 255).byte()
        back_mask = (back_mask * 255).byte()

        # Remove batch and channel dimensions (assuming batch size = 1 and channel = 1)
        mask_np = mask.squeeze().cpu().numpy()
        back_mask_np = back_mask.squeeze().cpu().numpy()

        # Convert to PIL Image and save
        mask_img = Image.fromarray(mask_np, mode='L')  # 'L' for grayscale
        back_mask_img = Image.fromarray(back_mask_np, mode='L')

        # Save images
        mask_img.save("./results_rafa/mask_image.png")
        back_mask_img.save("./results_rafa/back_mask_image.png")

    def save_nml_image(self, nmlF_normalized, nmlB_normalized):
        # Normalize to [0, 1] range for image representation
        nmlF_normalized = (nmlF_normalized + 1) / 2  # Map from [-1, 1] to [0, 1]
        nmlB_normalized = (nmlB_normalized + 1) / 2  # Map from [-1, 1] to [0, 1]

        # Convert to [0, 255] for image saving
        nmlF_normalized = (nmlF_normalized * 255).byte()
        nmlB_normalized = (nmlB_normalized * 255).byte()

        # Move tensors to CPU and convert to NumPy
        nmlF_np = nmlF_normalized.squeeze().permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        nmlB_np = nmlB_normalized.squeeze().permute(1, 2, 0).cpu().numpy()  # (H, W, 3)

        # Convert to PIL Image and save
        nmlF_img = Image.fromarray(nmlF_np, mode='RGB')  # Save as RGB image
        nmlB_img = Image.fromarray(nmlB_np, mode='RGB')

        # Save images
        nmlF_img.save("./results_rafa/nmlF_image.png")
        nmlB_img.save("./results_rafa/nmlB_image.png")