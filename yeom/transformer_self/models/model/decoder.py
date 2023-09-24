import copy
import torch.nn as nn


class Decoder(nn.Module):

    def __init__(self, decoder_block, n_layer, norm):
        super(Decoder, self).__init__()
        self.n_layer = n_layer
        self.layers = nn.ModuleList([copy.deepcopy(decoder_blcok)]
                                    for _ in range(self.n_layer))
        self.norm = norm

    def forward(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        # tgt_mask: Decoder의 input으로 주어지는 target sentence의 pad masking과 subsequent masking ( Self-Multi-Head Attention Layer에서 사용)
        # src_tgt_mask: Self-Multi-Head Attention Layer에서 넘어온 query, Encoder에서 넘어온 key, value 사이의 pad masking이다.

        out = tgt
        for layer in self.layers:
            out = layer(out, encoder_out, tgt_mask, src_tgt_mask)
        out = self.norm(out)
        return out
