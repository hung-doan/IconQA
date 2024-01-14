from torch import nn
from .Attention import SelfAttention


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        mlp_dim,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
    ):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend(
                [
                    Residual(
                        PreNormDrop(
                            dim,
                            dropout_rate,
                            SelfAttention(
                                dim, heads=heads, dropout_rate=attn_dropout_rate
                            ),
                        )
                    ),
                    Residual(
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout_rate))
                    ),
                ]
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LayerBlock(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        mlp_dim,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
    ):
        super().__init__()
        # Att
        self.att = SelfAttention(
                                dim, heads=heads, dropout_rate=attn_dropout_rate
                            )
        self.att_norm = nn.LayerNorm(dim)
        self.att_norm_drop = nn.Dropout(p=dropout_rate)

        # FF
        self.ffn = FeedForward(dim, mlp_dim, dropout_rate)
        self.ffn_norm = nn.LayerNorm(dim)
        

    def forward(self, x):

        # Attention Block
        # Norm > Att > Residual 
        residual = x
        x = self.att_norm(x)
        x, att_weights = self.att(x)
        x = self.att_norm_drop(x)
        x = x + residual

        # Feed forward block
        # Norm > FFN
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + residual
        
        return x, att_weights
class TransformerModelV2(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        mlp_dim,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
    ):
        super().__init__()
        self.layers = []
        for _ in range(depth):
            layer = LayerBlock(dim, heads, mlp_dim, dropout_rate, attn_dropout_rate)
            self.layers.append(layer)

    def forward(self, x):
        att_weights = []
        for layer in self.layers:
            x, layer_att_weights = layer(x)
            att_weights.append(layer_att_weights)
        return x, att_weights
