import torch
from LDRNet import LDRNet
from dataloader import CardDataset
from torch.utils.data import DataLoader
from loss import LineLoss, regression_loss
import numpy as np
from tqdm import tqdm
import yaml
import argparse
from torch.optim import Adam

def Loss(config, model, x, y, training, coord_size=8, class_list=[1], use_line_loss = True):
    weighted_loc_loss = regression_loss()
    line_loss = LineLoss()
    if coord_size>8:
        assert coord_size%4 == 0, "Coord Size Wrong"
        size_per_line = int((coord_size-8)/4 /2)
        coord_start = y[:, 0:8]
        coord_end = torch.cat([y[:, 2:8], y[:, 0:2]], axis = 1)
        coord_increment = (coord_end - coord_start)/(size_per_line + 1)
        new_coord = coord_start + coord_increment
        for index in range(1, size_per_line):
            new_coord = torch.cat([new_coord, coord_start + (index + 1)*coord_increment], axis = 1)
            y = torch.cat([new_coord, y[:, 8]], axis = 1)
    corner_y_, border_y_, class_y_ = model(x, training=training)
    coord_y_ = torch.cat([corner_y_, border_y_], axis=1)
    coord_y = y[:, 0:coord_size]  
    y__end = coord_size
    losses = []
    total_loss = 0
    for class_size in class_list:
        class_y = y[:, y_end]
        y_end += 1
        y__end += class_size + 1
        class_loss = torch.nn.BCEWithLogitsLoss(class_y, class_y)
        losses.append(class_loss)
        total_loss += class_loss
    loc_loss=config['loss']['loss_ratio']*weighted_loc_loss(coord_y, coord_y_, weights=1)
    total_loss+=loc_loss*config['loss']['class_loss_ratio']
    losses.append(loc_loss*config['loss']['class_loss_ratio'])

    if coord_size>8:
        total_slop_loss = 0
        total_diff_loss = 0
        for index in range(4):
            line = coord_y_[:, index * 2:index * 2 + 2]
            for coord_index in range(size_per_line):
                line = torch.cat(
                    [line, coord_y_[:, 8 + coord_index * 8 + index * 2:8 + coord_index * 8 + index * 2 + 2]], axis=1)
                # liner = tf.concat([liner,coord_y[:,8+coord_index*8+index*2:8+coord_index*8+index*2+2]],axis=1)
            line = torch.cat([line, coord_y_[:, (index * 2 + 2) % 8:(index * 2 + 2 + 2) % 8]], axis=1)
            cur_slop_loss, cur_diff_loss = line_loss(line)
        if use_line_loss:
            losses.append(total_slop_loss * config["loss"]["slop_loss_ratio"])
            losses.append(total_diff_loss * config["loss"]["diff_loss_ratio"])
            total_loss += total_slop_loss * config["loss"]["slop_loss_ratio"]
            total_loss += total_diff_loss * config["loss"]["diff_loss_ratio"]
        else:
            losses.append(total_slop_loss - total_slop_loss)  # total_slop_loss * slop_loss_ratio)
            losses.append(total_diff_loss - total_diff_loss)  # total_diff_loss * diff_loss_ratio)
        total_loss += 0  # total_slop_loss * slop_loss_ratio
        total_loss += 0  # total_diff_loss * diff_loss_ratio

    return total_loss, losses, [coord_y_, class_y_]


def train(ops):
    batch_size = ops.batch_size
    epochs = ops.batch_size
    label_path = ops.label_path
    img_path = ops.img_path
    class_size = ops.class_list
    dataset = CardDataset(label_path, img_folder=img_path, class_sizes=class_size, batch_size=batch_size)
    train_loader = DataLoader(dataset)
    optimizer = Adam()
    LDR = LDRNet()
    for epoch in tqdm(range(epochs)):
        step = 0
        for x, y in train_loader:
            step += 1
            loss, _, _ = Loss(ops, LDR, x, y ,True, ops['points_size']*2, class_list=ops['class_list'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if(step % 100):
                print(f"Current Epoch:{epoch}, Current Step:{step}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='/home/sasuke/repos/LDR_NET/config.yml', type=str)
    args = parser.parse_args()
    config = yaml.load(open(args.config_file, 'rb'))
    train(config)