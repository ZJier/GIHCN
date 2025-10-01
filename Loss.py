import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device='cuda', temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())
        
    def forward(self, emb_i, emb_j):
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        # loss = (2 * self.batch_size) / torch.sum(loss_partial)
        return loss


class JS_Loss(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.KL = nn.KLDivLoss(reduction='batchmean')
        self.sx = nn.Softmax(dim=1)
        self.log_sx = nn.LogSoftmax(dim=1)
    def forward(self, output_fea_js_1, output_fea_js_2):
        M_sx = (self.sx(output_fea_js_1) + self.sx(output_fea_js_2)) * 0.5
        P_Ls = self.log_sx(output_fea_js_1)
        P_Hs = self.log_sx(output_fea_js_2)
        loss_JS = (self.KL(P_Ls, M_sx) + self.KL(P_Hs, M_sx)) * 0.5
        return loss_JS


# Usage of Loss during the training process.
# ============================================================================
output_fea_con, output_cls = net(data, data2)
b_fea, c_fea = output_fea_con.shape
# ContrastiveLoss
output_fea_1 = output_fea_con[0:b_fea // 2, :]
output_fea_2 = output_fea_con[b_fea // 2:, :]
con_loss = ContrastiveLoss(b_fea // 2)
loss_view = con_loss.forward(output_fea_1, output_fea_2)
# CrossEntropyLoss
output_cls_1 = output_cls[0:b_fea // 2, :]
output_cls_2 = output_cls[b_fea // 2:, :]
loss_cls_1 = criterion(output_cls_1, target)
loss_cls_2 = criterion(output_cls_2, target)
# JS Divergence
output_fea_js_1 = output_fea_con[0:b_fea // 2, :]
output_fea_js_2 = output_fea_con[b_fea // 2:, :]
jss_loss = JS_Loss(b_fea // 2)
loss_JS = jss_loss.forward(output_fea_js_1, output_fea_js_2)
# Multi-Loss
loss = 1*(1*loss_view + 1*loss_JS) + 1*(loss_cls_1 + loss_cls_2)
# ============================================================================