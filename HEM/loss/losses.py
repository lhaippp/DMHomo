import math
import torch

import numpy as np
import torch.nn as nn

from kornia.geometry.linalg import transform_points


class LossL1(nn.Module):

    def __init__(self, reduction='mean'):
        super(LossL1, self).__init__()
        self.loss = nn.L1Loss(reduction=reduction)

    def __call__(self, input, target):
        return self.loss(input, target)


class LossL2(nn.Module):

    def __init__(self):
        super(LossL2, self).__init__()
        self.loss = nn.MSELoss()

    def __call__(self, input, target):
        return self.loss(input, target)


class Mask_Loss(nn.Module):

    def __init__(self, weight=(1, 1)):
        super(Mask_Loss, self).__init__()
        self.weight = weight
        self.loss = nn.BCELoss()

    def gen_weight(self, h, w):
        interval = self.weight[1] - self.weight[0]
        weight = interval * torch.arange(h) / h + self.weight[0] - 1
        weight = torch.repeat_interleave(weight, w)
        return weight.view(1, 1, h, w)

    def __call__(self, x):
        bs, _, h, w = x.size()
        weight = self.gen_weight(h, w)
        weight = weight.repeat(bs, 1, 1, 1).to(x.device)
        mask_loss = self.loss(x, weight)
        return mask_loss


class NLLLaplace:
    """ Computes Negative Log Likelihood loss for a (single) Laplace distribution. """

    def __init__(self, reduction='mean', ratio=1.0):
        """
        Args:
            reduction: str, type of reduction to apply to loss
            ratio:
        """
        super().__init__()
        self.reduction = reduction
        self.ratio = ratio

    def __call__(self, gt_flow, est_flow, log_var, mask=None):
        """
        Args:
            gt_flow: ground-truth flow field, shape (b, 2, H, W)
            est_flow: estimated flow field, shape (b, 2, H, W)
            log_var: estimated log variance, shape (b, 1, H, W)
            mask: valid mask, where the loss is computed. shape (b, 1, H, W)
        """
        b, _, h, w = gt_flow.shape
        loss1 = math.sqrt(2) * torch.exp(-0.5 * log_var) * \
            torch.abs(gt_flow - est_flow)
        # each dimension is multiplied
        loss2 = 0.5 * log_var
        loss = loss1 + loss2

        if mask is not None:
            mask = ~torch.isnan(loss.detach()) & ~torch.isinf(
                loss.detach()) & mask
        else:
            mask = ~torch.isnan(loss.detach()) & ~torch.isinf(loss.detach())

        if torch.isnan(loss.detach()).sum().ge(1) or torch.isinf(
                loss.detach()).sum().ge(1):
            print('mask or inf in the loss ! ')

        if self.reduction == 'mean':
            if mask is not None:
                loss = torch.masked_select(loss, mask).mean()
            else:
                loss = loss.mean()
            return loss
        elif 'weighted_sum' in self.reduction:
            if mask is not None:
                loss = loss * mask.float()
                L = 0
                for bb in range(0, b):
                    norm_const = float(h) * float(w) / \
                        (mask[bb, ...].sum().float() + 1e-6)
                    L += loss[bb][mask[bb, ...] != 0].sum() * norm_const
                if 'normalized' in self.reduction:
                    return L / b
                return L

            if 'normalized' in self.reduction:
                return loss.sum() / b
            return loss
        else:
            raise ValueError


def triplet_loss(a, p, n, margin=1.0, exp=1, reduce=False, size_average=False):
    triplet_loss = nn.TripletMarginLoss(margin=margin,
                                        p=exp,
                                        reduce=reduce,
                                        size_average=size_average)
    return triplet_loss(a, p, n)


def compute_losses(data, endpoints, params):
    loss = {}

    flow_b_gt, flow_f_gt = data["flow_gt_patch"][:, :2, :, :], data[
        "flow_gt_patch"][:, 2:, :, :]
    flow_b, flow_f = endpoints["flow_b"], endpoints["flow_f"]
    mask_b, mask_f = endpoints["mask_b"], endpoints["mask_f"]
    if params.normalize_mask:
        # print('warp and normalize the mask')
        mask_b = mask_f = endpoints["mask_fusion"]
    fil_features = endpoints["fil_features"]

    # Loss Definition
    pl_criterion = LossL1(reduction='mean')
    mask_loss = Mask_Loss()
    nll_laplace = NLLLaplace(reduction='mean')

    # loss["triplet"] = triplet_loss(
    #     fil_features["img1_patch_fea"], fil_features["img2_patch_fea_warp"], fil_features["img2_patch_fea"]).mean() + triplet_loss(
    #         fil_features["img2_patch_fea"], fil_features["img1_patch_fea_warp"], fil_features["img1_patch_fea"]).mean()
    loss["unsup"] = params.unsup_loss_weight * (
        pl_criterion(mask_f * fil_features["img1_patch_fea"],
                     mask_f * fil_features["img2_patch_fea_warp"]) +
        pl_criterion(mask_b * fil_features["img2_patch_fea"],
                     mask_b * fil_features["img1_patch_fea_warp"]))

    loss['mask_reg'] = params.mask_reg_loss_weight * \
        (mask_loss(mask_b) + mask_loss(mask_f))
    loss['nll'] = params.mask_nll_loss_weight * \
        (nll_laplace(flow_f_gt, flow_f, (1 - mask_f)) +
         nll_laplace(flow_b_gt, flow_b, (1 - mask_b)))

    loss['supervise'] = params.sup_loss_weight * (
        pl_criterion(mask_b * flow_b, mask_b * flow_b_gt) +
        pl_criterion(mask_f * flow_f, mask_f * flow_f_gt))
    # loss['supervise'] = params.sup_loss_weight * (pl_criterion(flow_b, flow_b_gt) + pl_criterion(flow_f, flow_f_gt))
    # loss['supervise'] = pl_criterion(flow_b, flow_b_gt) + pl_criterion(flow_f, flow_f_gt)

    loss['total'] = loss['supervise'] + \
        loss['mask_reg'] + loss["unsup"] + loss['nll']
    # print(f"gpu:{torch.cuda.current_device()} loss:{loss['total']}")
    return loss


def compute_metrics(data, output, manager):
    metrics = {}
    with torch.no_grad():
        # compute metrics
        B = data["label"].size()[0]
        outputs = np.argmax(output["p"].detach().cpu().numpy(), axis=1)
        accuracy = np.sum(
            outputs.astype(np.int32) ==
            data["label"].detach().cpu().numpy().astype(np.int32)) / B
        metrics['accuracy'] = accuracy
        return metrics


# def ComputeErrH(src, dst, H):
#     src = src.unsqueeze(0).unsqueeze(0).detach().cpu().numpy().astype('float64')
#     dst = dst.detach().cpu().numpy().astype('float64')
#     src_warp = cv2.perspectiveTransform(src, H.squeeze(0).detach().cpu().numpy()).reshape(2)
#     err = np.sqrt((dst[0] - src_warp[0])**2 + (dst[1] - src_warp[1])**2)
#     return err


def ComputeErrH_kornia(src, dst, H):
    # src,dst:(B, N, 2)
    # H:(N, 3, 3)
    src_warp = transform_points(H, src)
    err = torch.sqrt((src_warp[:, :, 0] - src[:, :, 0])**2 +
                     (src_warp[:, :, 1] - dst[:, :, 1])**2)
    return err


def ComputeErrH_v2(src, dst, H):
    '''
    :param src: B, N, 2
    :param dst: B, N, 2
    :param H: B, 3, 3
    '''
    src, dst = src.unsqueeze(0).unsqueeze(0), dst.unsqueeze(0).unsqueeze(0)
    src_warp = transform_points(H.unsqueeze(0), src)
    err = torch.linalg.norm(dst - src_warp)
    return err


def ComputeErrFlow(src, dst, flow):
    src_t = src + flow[int(src[1]), int(src[0])]
    error = torch.linalg.norm(dst - src_t)
    return error


def ComputeErr(src, dst):
    error = torch.linalg.norm(dst - src)
    return error


# def compute_eval_results(data_batch, output_batch, accelerator):
#     imgs_full = data_batch["imgs_gray_full"]
#     accelerator.print(f"imgs_full shape {imgs_full.shape}")

#     # pt_set = list(map(eval, data_batch["pt_set"]))
#     pt_set = data_batch["pt_set"]
#     accelerator.print(f"pt_set shape {pt_set.shape}")
#     # pt_set = list(map(lambda x: x['matche_pts'], pt_set))

#     batch_size, _, img_h, img_w = imgs_full.shape
#     Homo_b = output_batch["Homo_b"]
#     Homo_f = output_batch["Homo_f"]
#     accelerator.print(f"Homo_b shape {Homo_b.shape}")

#     src = pt_set[:, :6, 0]
#     dst = pt_set[:, :6, 1]

#     # errs_m = []
#     # pred_errs = []
#     # for i in range(batch_size):
#     #     # pts = torch.Tensor(pt_set[i]).to(accelerator.device)
#     #     pts = pt_set[i]
#     #     err = 0
#     #     for j in range(6):
#     #         p1 = pts[j][0]
#     #         p2 = pts[j][1]
#     #         src, dst = p1, p2
#     #         pred_err = min(ComputeErrH(src=src, dst=dst, H=Homo_b[i]), ComputeErrH(src=dst, dst=src, H=Homo_f[i]))
#     #         err += pred_err
#     #         pred_errs.append(pred_err)
#     #     err /= 6
#     #     errs_m.append(err)

#     # errs_b,errs_b:(B, 1)
#     errs_b = torch.mean(ComputeErrH_kornia(src, dst, Homo_b), 1)
#     errs_f = torch.mean(ComputeErrH_kornia(dst, src, Homo_f), 1)

#     for i, _ in enumerate(errs_b):
#         errs_b[i] = torch.minimum(errs_b[i], errs_f[i])
#     # eval_results = {"errors_m": np.array(errs_m)}
#     accelerator.print(f"eval_results {errs_b}")
#     return errs_b


def compute_eval_results(data_batch, output_batch):
    imgs_full = data_batch["imgs_gray_full"]
    device = imgs_full.device
    batch_size, _, img_h, img_w = imgs_full.shape

    pt_set = data_batch["pt_set"]
    flow_f = output_batch["flow_f"]
    flow_b = output_batch["flow_b"]
    # print(f"flow_f shape {flow_f.shape}")

    # for unit test
    # for i in range(flow_f.shape[0]):
    #     flow_f[i] = torch.zeros_like(flow_f[i])
    #     flow_b[i] = torch.zeros_like(flow_b[i])

    errs_m = []
    pred_errs = []
    for i in range(batch_size):
        # pts = torch.from_numpy(pt_set[i]).to(device)
        pts = pt_set[i]
        err = 0
        for j in range(6):
            p1 = pts[j][0]
            p2 = pts[j][1]
            src, dst = p1, p2
            pred_err = min(ComputeErrFlow(src=src, dst=dst, flow=flow_f[i]),
                           ComputeErrFlow(src=dst, dst=src, flow=flow_b[i]))
            err += pred_err
            pred_errs.append(pred_err)
        err /= 6

        errs_m.append(err)

    return errs_m
