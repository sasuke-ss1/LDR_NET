import torch
import torch.nn as nn


class regression_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred, weights):
        mse = torch.mean(((y_pred - y_true) ** 2) * weights, 1)
        return mse


class LineLoss(nn.Module):
    def __init__(self, Beta, Gamma):
        super().__init__()
        self.beta = Beta
        self.gamma = Gamma

    def forward(self, line):
        line_x = line[:, 0::2]
        line_y = line[:, 1::2]

        x_diff = line_x[:, 1:] - line_x[:, 0:-1]
        y_diff = line_y[:, 1:] - line_y[:, 0:-1]

        x_diff_start = x_diff[:, 1:]
        x_diff_end = x_diff[:, 0:-1]
        y_diff_start = y_diff[:, 1:]
        y_diff_end = y_diff[:, 0:-1]

        n = 4 * line.shape[1]
        similarity = (x_diff_start * x_diff_end + y_diff_start * y_diff_end) / (
            torch.sqrt(torch.square(x_diff_start) + torch.square(y_diff_start) + 0.0000000000001)
            * torch.sqrt(torch.square(x_diff_end) + torch.square(y_diff_end))
            + 0.0000000000001  # to avoid division by zero
        )

        slop_loss = torch.sum(1 - similarity, axis=1) / (n - 4)

        x_diff_loss = torch.abs(torch.abs(x_diff[:, 1:]) - torch.abs(x_diff[:, 0:-1]))
        y_diff_loss = torch.abs(torch.abs(y_diff[:, 1:]) - torch.abs(y_diff[:, 0:-1]))
        distance_loss = torch.sum(x_diff_loss + y_diff_loss, axis=1) / (n - 4)

        line_loss = self.beta * slop_loss + self.gamma * distance_loss

        return line_loss
