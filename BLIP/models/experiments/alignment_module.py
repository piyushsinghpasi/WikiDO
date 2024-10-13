import torch.nn as nn
import torch
from models.attention import MultiHeadedAttention

class alignment_block(nn.Module):
    
    def __init__(self, d_model) -> None:
        super(alignment_block, self).__init__()
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # self.self_attention = nn.MultiheadAttention(d_model, num_heads=1, dropout=0.2, batch_first=True)
        # self.cross_attention = nn.MultiheadAttention(d_model, num_heads=1, dropout=0.2, batch_first=True)

        self.self_attention = MultiHeadedAttention(n_feat=d_model, n_head=8, dropout_rate=0.2, proj_layers=['query', 'key', 'value', 'final'])
        self.cross_attention = MultiHeadedAttention(n_feat=d_model, n_head=8, dropout_rate=0.2, proj_layers=['query', 'key', 'value', 'final'])

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.ReLU(),
            nn.Linear(4*d_model, d_model)
        )
        
    def forward(self, x, x_cross, mask=None, print_wts=False, save_path=None, self_mask=None, cross_mask=None, skip_self=False, skip_x=False):
        
        if not skip_self:
            x_norm1 = self.norm1(x)
            x_norm1, wt_self = self.self_attention(x_norm1,x_norm1,x_norm1, mask=self_mask)
            x = x + x_norm1
        
        x_norm2 = self.norm2(x)
        x_norm2, wt_cross = self.cross_attention(x_norm2,x_cross,x_cross, mask=cross_mask)

        if skip_x:
            x = x_norm2
        else:
            x = x + x_norm2
        
        x_norm3 = self.norm3(x)
        x_norm3 = self.feed_forward(x_norm3)
        x = x + x_norm3
        
        if print_wts:
            if not skip_self:
                print("self", wt_self)
            print("cross", wt_cross)

        if save_path:

            if not skip_self:
                self_path = f"{save_path}_self.pt"
                torch.save(wt_self.detach().cpu(), self_path)
                print(f"Self Attn saved at: {self_path}")
            
            cross_path = f"{save_path}_cross.pt"            
            torch.save(wt_cross.detach().cpu(), cross_path)
            print(f"Cross Attn saved at: {cross_path}")

        return x
    
class multi_alignment_block(nn.Module):
    
    def __init__(self, d_model, num_layers, **kwargs) -> None:
        super(multi_alignment_block, self).__init__()
        
        self.block = nn.ModuleList([alignment_block(d_model) for _ in range(num_layers)])
        self.update_x_cross = kwargs["update_x_cross"]
        self.swap_attn_inps = kwargs["swap_attn_inps"]
        
    def forward(self, x, x_cross, mask=None, print_wts=False, save_path=None):
        
        for idx, layer in enumerate(self.block):
            if print_wts:
                print("idx", idx)
            if save_path:
                save_path_idx = f"{save_path}_layer{idx}"
            else:
                save_path_idx = save_path

            if self.update_x_cross:
                if self.swap_attn_inps:
                    x_cross = layer(x_cross, x, cross_mask=mask, print_wts=print_wts, save_path=save_path_idx)
                else:
                    x_cross = layer(x, x_cross, self_mask=mask, print_wts=print_wts, save_path=save_path_idx)
            else:
                x = layer(x, x_cross, self_mask=mask, print_wts=print_wts, save_path=save_path_idx)
            

        if self.update_x_cross:
            return x_cross
        
        return x
        
class residual_alignment_block(nn.Module):
    
    def __init__(self, d_model, num_layers, **kwargs) -> None:
        super(residual_alignment_block, self).__init__()
        
        self.block = nn.ModuleList([alignment_block(d_model) for _ in range(num_layers)])
        self.skip_x = kwargs["skip_x"]
        self.retain_org_x = kwargs["retain_org_x"]
        
    def forward(self, x, x_cross, mask=None, print_wts=False, save_path=None):
        x_new = None
        x_res = None
        x_final = torch.zeros_like(x)
        for idx, layer in enumerate(self.block):
            if print_wts:
                print("idx", idx)
            if save_path:
                save_path_idx = f"{save_path}_layer{idx}"
            else:
                save_path_idx = save_path
            
            if idx==0 or self.retain_org_x:
                x_res = x
            else:
                x_res = x-x_new
            x_new = layer(x_res, x_cross, self_mask=mask, print_wts=print_wts, save_path=save_path_idx, skip_self=True, skip_x=self.skip_x)
            x_final = x_final + x_new
            
        return x_final
        
        