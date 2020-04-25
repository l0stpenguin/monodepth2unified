import torch


def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def compute_pairwise_brightness_loss(pred, target):
    abs_diff = charbonnier_penalty(torch.abs(target - pred))
    l1_loss = abs_diff.mean(1, True)
    return l1_loss


def compute_pairwise_gradient_loss(pred, target):
    grad_pred_x = pred[:, :, :, :-1] - pred[:, :, :, 1:]
    grad_pred_y = pred[:, :, :-1, :] - pred[:, :, 1:, :]

    grad_tgt_x = target[:, :, :, :-1] - target[:, :, :, 1:]
    grad_tgt_y = target[:, :, :-1, :] - target[:, :, 1:, :]

    x_diff = charbonnier_penalty(torch.abs(grad_pred_x - grad_tgt_x))
    y_diff = charbonnier_penalty(torch.abs(grad_pred_y - grad_tgt_y))

    abs_diff = torch.cat((x_diff, x_diff[:, :, :, -1:]), 3) + \
               torch.cat((y_diff, y_diff[:, :, -1:, :]), 2)
    l1_loss = abs_diff.mean(1, True)
    return l1_loss


def compute_pairwise_geometry_loss(computed_depth, sampled_depth):
    return charbonnier_penalty(torch.abs((computed_depth - sampled_depth)))


def charbonnier_penalty(loss):
    return torch.sqrt(loss ** 2 + 0.001 ** 2)