import torch
from LDRNet import LDRNet
from dataloader import DocData
from torch.utils.data import DataLoader
from loss import LineLoss, regression_loss
import numpy as np
from torchvision import transforms
from tqdm import tqdm
import yaml
import argparse
from torch.optim import Adam

def Loss(config, model, x, y, training, coord_size=8, class_list=[1], use_line_loss = True):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    weighted_loc_loss = regression_loss()
    line_loss = LineLoss(0.5, 0.5) # Random Values
    if coord_size>8:
        assert coord_size%4 == 0, "Coord Size Wrong"
        size_per_line = int((coord_size-8)/4 /2)
        coord_start = y[:, 0:8]
        coord_end = torch.cat([y[:, 2:8], y[:, 0:2]], axis = 1)
        coord_increment = (coord_end - coord_start)/(size_per_line + 1)
        new_coord = coord_start + coord_increment
        for index in range(1, size_per_line):
            new_coord = torch.cat([new_coord, coord_start + (index + 1)*coord_increment], axis = 1)
            #print(y.shape)
        y = torch.cat([new_coord, y[:, 0:8]], axis = 1)
    corner_y_, border_y_, class_y_ = model(x)
    coord_y_ = torch.cat([corner_y_, border_y_], axis=1)
    coord_y = y[:, 0:coord_size]  
    y_end = coord_size
    y__end = coord_size
    losses = []
    total_loss = 0
    for class_size in class_list:
        class_y = torch.ones((y.shape[0], 1), device=device)
        y_end += 1
        y__end += class_size + 1
        #print(class_y)
        #print(class_y_)
        bce = torch.nn.BCEWithLogitsLoss(reduction='none')
        class_loss = bce(class_y, class_y_)
        losses.append(class_loss)
        total_loss += class_loss.squeeze(1)
    loc_loss=config['loss']['loss_ratio']*weighted_loc_loss(coord_y, corner_y_, weights=1)
    #print(total_loss.shape)
    #print(loc_loss.shape)
    total_loss+=loc_loss*config['loss']['class_loss_ratio']
    losses.append(loc_loss*config['loss']['class_loss_ratio'])

    if coord_size>8:
        total_line_loss = 0
        for index in range(4):
            line = coord_y_[:, index * 2:index * 2 + 2]
            for coord_index in range(size_per_line):
                line = torch.cat(
                    [line, coord_y_[:, 8 + coord_index * 8 + index * 2:8 + coord_index * 8 + index * 2 + 2]], axis=1)
                # liner = tf.concat([liner,coord_y[:,8+coord_index*8+index*2:8+coord_index*8+index*2+2]],axis=1)
            line = torch.cat([line, coord_y_[:, (index * 2 + 2) % 8:(index * 2 + 2 + 2) % 8]], axis=1)
            line_loss_ = line_loss(line)
            if use_line_loss:
                total_line_loss = total_line_loss + line_loss_
            
        losses.append(total_line_loss)
        #else:
      #      losses.append(total_slop_loss - total_slop_loss)  # total_slop_loss * slop_loss_ratio)
       #     losses.append(total_diff_loss - total_diff_loss)  # total_diff_loss * diff_loss_ratio)
    
    total_loss += total_line_loss
    return total_loss, losses, [coord_y_, class_y_]


def train(ops):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = ops['batch_size']
    epochs = ops['epochs']
    img_path = ops['img_folder_path']
    class_size = ops['class_list']
    Transform = transforms.Compose([transforms.ToTensor()])
    dataset = DocData(img_dir=img_path, transforms=Transform)
    train_loader = DataLoader(dataset, batch_size=batch_size)
    LDR = LDRNet(points_size=ops['points_size']).to(device)
    optimizer = Adam(LDR.parameters())
    for epoch in range(epochs):
        step = 0
        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)
            #print(x.shape)
            step += 1
            loss, _, _ = Loss(ops, LDR, x, y ,True, ops['points_size']*2, class_list=ops['class_list'])
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
            if not (step % 20):
                print(f"Current Epoch:{epoch+1}, Current Step:{step}, Current Loss: {loss.mean()}")
        if not (epoch+1 % 20):
            torch.save(LDR, "./model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='/home/sasuke/repos/LDR_NET/config.yml', type=str)
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config_file, "r"))
    #print(config)
    train(config)
