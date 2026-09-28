import torch.nn as nn
import torch
import torch.nn.functional as F
from pathlib import Path


class VGGEncoder(nn.Module):
    def __init__(self, vgg_path):
        super(VGGEncoder, self).__init__()

        vgg_path = Path(vgg_path)
        if not vgg_path.exists():
            raise FileNotFoundError(
                f"VGG weights not found at {vgg_path}. "
                f"Place vgg_normalised.pth there, or pass the correct path."
            )

        self.vgg = nn.Sequential(
            nn.Conv2d(3, 3, (1, 1)),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 64, (3, 3)),
            nn.ReLU(),  # relu1-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),  # relu1-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),  # relu2-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),  # relu2-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU(),  # relu3-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, (3, 3)),
            nn.ReLU(),  # relu4-1, this is the last layer used
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU()  # relu5-4
        )

        # map_location keeps this loadable on a CPU-only machine even if the
        # checkpoint was originally saved from a CUDA tensor; weights_only=False
        # is required because this checkpoint uses PyTorch's legacy (pre-2016)
        # pickle format, which newer PyTorch versions no longer treat as the default.
        state_dict = torch.load(str(vgg_path), map_location="cpu", weights_only=False)
        self.vgg.load_state_dict(state_dict)

        self.vgg = nn.Sequential(*list(self.vgg.children())[:31])
        enc_layers = list(self.vgg.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])
        self.enc_2 = nn.Sequential(*enc_layers[4:11])
        self.enc_3 = nn.Sequential(*enc_layers[11:18])
        self.enc_4 = nn.Sequential(*enc_layers[18:31])

        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    def forward(self, input, is_test=False):
        h1 = self.enc_1(input)
        h2 = self.enc_2(h1)
        h3 = self.enc_3(h2)
        h4 = self.enc_4(h3)
        if is_test:
            return h4
        return h1, h2, h3, h4


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, (3, 3)),
        )

    def forward(self, input):
        return self.net(input)

    def load_pretrained(self, decoder_path, map_location="cpu"):
        """
        Load trained decoder weights (e.g. decoder.pth from the original
        AdaIN release). Without this, the decoder's weights are randomly
        initialised and will produce noise, not images — the decoder is
        the one part of this pipeline that actually needs training data
        to be useful, unlike VGGEncoder which is used exactly as published.
        """
        decoder_path = Path(decoder_path)
        if not decoder_path.exists():
            raise FileNotFoundError(f"Decoder weights not found at {decoder_path}.")
        state_dict = torch.load(str(decoder_path), map_location=map_location, weights_only=False)
        self.load_state_dict(state_dict)


# The missing piece: Adaptive Instance Normalization (Huang & Belongie, 2017,
# arXiv:1703.06868). VGGEncoder turns images into features and Decoder turns
# features back into images, but nothing previously connected the two for an
# actual style transfer. This is that connection.

def calc_mean_std(feat, eps=1e-5):
    """
    Per-channel mean and standard deviation of a (N, C, H, W) feature map,
    computed over the spatial dimensions only. This is the statistic AdaIN
    aligns between content and style features.
    """
    size = feat.size()
    assert len(size) == 4, f"Expected a 4D (N, C, H, W) tensor, got shape {tuple(size)}"
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    """
    The core AdaIN operation: normalise the content feature map to zero
    mean / unit variance (per channel), then rescale and re-center it using
    the style feature map's own mean and variance. This is what makes the
    output take on the style image's color/texture statistics while keeping
    the content image's spatial structure.
    """
    assert content_feat.size()[:2] == style_feat.size()[:2], (
        "Content and style features must share the same batch size and channel count."
    )
    content_mean, content_std = calc_mean_std(content_feat)
    style_mean, style_std = calc_mean_std(style_feat)

    normalized_content = (content_feat - content_mean) / content_std
    return normalized_content * style_std + style_mean


def calc_content_loss(input_feat, target_feat):
    """MSE between generated and target relu4_1 features (used when training the decoder)."""
    return F.mse_loss(input_feat, target_feat)


def calc_style_loss(input_feat, target_feat):
    """
    Style loss as defined in the AdaIN paper: match mean and standard
    deviation between generated and style features at one layer.
    Sum this across relu1_1..relu4_1 for the full multi-layer style loss.
    """
    input_mean, input_std = calc_mean_std(input_feat)
    target_mean, target_std = calc_mean_std(target_feat)
    return F.mse_loss(input_mean, target_mean) + F.mse_loss(input_std, target_std)


class StyleTransferModel(nn.Module):
    """
    End-to-end AdaIN style transfer: VGGEncoder -> AdaIN -> Decoder.

    This is the Python/PyTorch equivalent of what Restyle's web app runs via
    TensorFlow.js in the browser — same theory (Section 4.4), same four-layer
    encoder, same AdaIN operation. Useful for local experimentation, generating
    comparison figures, or as a reference implementation to train your own
    decoder against.

    Parameters
    ----------
    vgg_path : str or Path
        Path to vgg_normalised.pth.
    decoder_path : str or Path, optional
        Path to a pretrained decoder.pth. If omitted, the decoder is randomly
        initialised and stylize() will produce noise, not images — training
        or loading trained decoder weights is required for real output.
    device : torch.device or str, optional
        Defaults to CUDA if available, else CPU.
    """

    def __init__(self, vgg_path, decoder_path=None, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = VGGEncoder(vgg_path).to(self.device).eval()
        self.decoder = Decoder().to(self.device)
        if decoder_path is not None:
            self.decoder.load_pretrained(decoder_path, map_location=self.device)
        else:
            print(
                "Warning: no decoder_path given. StyleTransferModel.decoder has "
                "random weights and stylize() will not produce a meaningful image "
                "until you either load a pretrained decoder or train one."
            )
        self.decoder.eval()

    @torch.no_grad()
    def stylize(self, content, style, alpha=1.0):
        """
        Stylize a content image using a style image.

        Parameters
        ----------
        content : torch.Tensor
            (1, 3, H, W) content image tensor, already normalised to [0, 1].
        style : torch.Tensor
            (1, 3, H, W) style image tensor, same format.
        alpha : float, default 1.0
            Style strength, from 0.0 (returns the content features
            unchanged, i.e. the original photo) to 1.0 (full stylization).
            This is the exact same control as the "style strength" slider
            in Restyle's web app, implemented here at the feature level
            instead of by blending finished pixels.

        Returns
        -------
        torch.Tensor
            (1, 3, H, W) stylized image tensor.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be between 0.0 and 1.0, got {alpha}")

        content = content.to(self.device)
        style = style.to(self.device)

        content_feat = self.encoder(content, is_test=True)
        style_feat = self.encoder(style, is_test=True)

        stylized_feat = adaptive_instance_normalization(content_feat, style_feat)
        # alpha-blend in feature space between the untouched content features
        # and the fully AdaIN-normalized features, then decode once.
        blended_feat = alpha * stylized_feat + (1 - alpha) * content_feat

        return self.decoder(blended_feat).clamp(0, 1)