import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def pitchyaw2xyz(pitchyaw):
    pitches, yaws = pitchyaw[:, 0], pitchyaw[:, 1]
    x = -torch.cos(pitches) * torch.sin(yaws)
    y = -torch.sin(pitches)
    z = -torch.cos(pitches) * torch.cos(yaws)
    result = torch.stack((x, y, z), dim=1)
    return result

# def my_pitchyaw2xyz(pitchyaw):
#     pitches, yaws = pitchyaw[:, 0], pitchyaw[:, 1]
#
#     x = torch.cos(pitches) * torch.sin(yaws)
#     y = torch.sin(pitches)
#     z = torch.cos(pitches) * torch.cos(yaws)
#
#     return torch.stack((x, y, z), dim=-1)

def angular_loss(vectors1, vectors2):
    cos_sims = torch.sum(vectors1 * vectors2, dim=1) / (torch.norm(vectors1, dim=1) * torch.norm(vectors2, dim=1))
    cos_sims = torch.clamp(cos_sims, -1.0, 1.0)
    angle_diffs = torch.acos(cos_sims).mean()
    angle_diffs = angle_diffs * (180 / torch.tensor(np.pi))

    # cos_similarity = nn.CosineSimilarity(dim=1)
    # cos_sims_nn = cos_similarity(vectors1, vectors2)
    # angle_diffs_nn = torch.acos(cos_sims_nn).mean()
    # angle_diffs_nn = angle_diffs_nn * (180 / torch.tensor(np.pi))
    #
    # assert torch.allclose(angle_diffs, angle_diffs_nn), "Values differ!"
    #
    # # If assertion fails, show value differences
    # if not torch.allclose(angle_diffs, angle_diffs_nn):
    #     print("Code you provided:", angle_diffs.item())
    #     print("torch.nn.CosineSimilarity:", angle_diffs_nn.item())

    return angle_diffs

# def my_angular_loss(predictions, targets):
#     predictions = F.normalize(predictions, dim=-1)
#     targets = F.normalize(targets, dim=-1)
#     loss = torch.acos(torch.sum(predictions * targets, dim=-1))
#     return loss.mean()
