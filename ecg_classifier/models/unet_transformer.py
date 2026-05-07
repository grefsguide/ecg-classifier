import torch
from torch import nn
import torch.nn.functional as functional


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2,
        )
        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_x != 0 or diff_y != 0:
            x = functional.pad(
                x,
                [
                    diff_x // 2,
                    diff_x - diff_x // 2,
                    diff_y // 2,
                    diff_y - diff_y // 2,
                ],
            )

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class TinyUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int,
    ) -> None:
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.inc = DoubleConv(in_channels, c1)
        self.down1 = DownBlock(c1, c2)
        self.down2 = DownBlock(c2, c3)
        self.down3 = DownBlock(c3, c4)

        self.up1 = UpBlock(c4, c3, c3)
        self.up2 = UpBlock(c3, c2, c2)
        self.up3 = UpBlock(c2, c1, c1)

        self.out_conv = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.out_conv(x)


class UnetSeriesTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        num_signal_maps: int = 8,
        seq_len: int = 256,
        unet_base_channels: int = 32,
        transformer_d_model: int = 128,
        transformer_nhead: int = 8,
        transformer_num_layers: int = 4,
        transformer_ff_dim: int = 256,
        dropout: float = 0.1,
        softmax_temperature: float = 10.0,
    ) -> None:
        super().__init__()

        if transformer_d_model % transformer_nhead != 0:
            raise ValueError(
                "transformer_d_model must be divisible by transformer_nhead"
            )

        self.num_signal_maps = num_signal_maps
        self.seq_len = seq_len
        self.softmax_temperature = softmax_temperature

        self.unet = TinyUNet(
            in_channels=in_channels,
            out_channels=num_signal_maps,
            base_channels=unet_base_channels,
        )

        self.sequence_pool = nn.AdaptiveAvgPool1d(seq_len)
        self.input_projection = nn.Linear(num_signal_maps, transformer_d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            dim_feedforward=transformer_ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=transformer_num_layers,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, transformer_d_model))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, seq_len + 1, transformer_d_model)
        )

        self.head = nn.Sequential(
            nn.LayerNorm(transformer_d_model),
            nn.Linear(transformer_d_model, num_classes),
        )

    def _extract_series(self, signal_maps: torch.Tensor) -> torch.Tensor:
        """
        signal_maps: [B, K, H, W]
        returns:     [B, K, W]
        """
        batch_size, _, height, _ = signal_maps.shape

        probabilities = torch.softmax(
            signal_maps * self.softmax_temperature,
            dim=2,
        )

        y_grid = torch.linspace(
            -1.0,
            1.0,
            steps=height,
            device=signal_maps.device,
            dtype=signal_maps.dtype,
        ).view(1, 1, height, 1)

        series = torch.sum(probabilities * y_grid, dim=2)
        return series.view(batch_size, self.num_signal_maps, -1)

    def forward_features(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        signal_maps = self.unet(images)
        series = self._extract_series(signal_maps)
        series = self.sequence_pool(series)  # [B, K, seq_len]
        tokens = series.transpose(1, 2)      # [B, seq_len, K]
        tokens = self.input_projection(tokens)

        batch_size = tokens.size(0)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_token, tokens], dim=1)
        tokens = tokens + self.position_embedding[:, : tokens.size(1)]

        encoded = self.transformer(tokens)
        cls_representation = encoded[:, 0]
        return cls_representation, signal_maps

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        cls_representation, _ = self.forward_features(images)
        logits = self.head(cls_representation)
        return logits

    @torch.no_grad()
    def extract_signal_maps_and_series(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        signal_maps = self.unet(images)
        series = self._extract_series(signal_maps)
        series = self.sequence_pool(series)
        return signal_maps, series