import torch
import torch.nn as nn

class ViT(nn.Module):
    def __init__(self, image_size=84, patch_size=84, n_channels=4, embed_dim=32, action_space=5, n_layers=2, n_heads=16):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.n_patches = (image_size // patch_size) ** 2
        self.patch_dim = n_channels * patch_size * patch_size
        self.patch_size = patch_size
        self.patch_embedding = nn.Linear(self.patch_dim, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.n_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.encoder_layer = nn.TransformerEncoderLayer(embed_dim, n_heads, 4 * embed_dim, dropout=0, batch_first=True, norm_first=True, bias=False)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, n_layers)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim, bias=False),
            nn.Linear(embed_dim, action_space)
        )

    def image2patch(self, image):
        batch, channel, height, width = image.shape

        #image: batch, channel, height // patch_size, width // patch_size, patch_size, patch_size
        image = image.reshape(batch, channel, height // self.patch_size, width // self.patch_size, self.patch_size, self.patch_size)

        #image: batch, height // patch_size, width // patch_size, patch_size, patch_size, channel
        image = image.permute(0, 2, 3, 4, 5, 1)

        #image: batch, height // patch_size * width // patch_size, patch_size * patch_size * channel
        image = image.reshape(batch, height // self.patch_size * width // self.patch_size, self.patch_size * self.patch_size * channel)

        return image
    
    def forward(self, image):
        patch = self.image2patch(image)
        patch = self.patch_embedding(patch)
        batch_size, n_seq, embed_dim = patch.shape
        cls_tokens = self.cls_token.repeat((batch_size, 1, 1))
        patch = torch.cat((cls_tokens, patch), dim=1)
        patch += self.pos_embedding[:, : n_seq + 1]
        patch = self.encoder(patch)
        patch = patch[:, 0]
        patch = self.mlp_head(patch)
        return patch

model = ViT()
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_params)