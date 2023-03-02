import argparse
import os
import torch
import yaml
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import glob
from dataloader import DocData
from LDRNet import LDRNet
from loss import LineLoss, regression_loss


def Loss(config, model, x, y, training, coord_size=8, class_list=[1], use_line_loss=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weighted_loc_loss = regression_loss()
    line_loss = LineLoss(0.8, 0.6)  # Random Values

    if coord_size > 8:
        assert coord_size % 4 == 0, "Coord Size Wrong"
        size_per_line = int((coord_size - 8) / (4 * 2))
        coord_start = y[:, 0:8]
        # print(coord_start)
        coord_end = torch.cat([y[:, 2:8], y[:, 0:2]], axis=1)
        # print(coord_end)
        coord_increment = (coord_end - coord_start) / (size_per_line + 1)
        new_coord = coord_start + coord_increment
        for index in range(1, size_per_line):
            new_coord = torch.cat([new_coord, coord_start + (index + 1) * coord_increment], axis=1)
            # print(y.shape)
        y = torch.cat([new_coord, y[:, 0:8]], axis=1)

    corner_y_, border_y_, class_y_ = model(x)
    coord_y_ = torch.cat([corner_y_, border_y_], axis=1)
    coord_y = y[:, 0:8]
    y_end = coord_size
    y__end = coord_size
    # print(corner_y_)

    losses = []
    total_loss = 0
    for class_size in class_list:
        class_y = torch.ones((y.shape[0], 1), device=device)
        y_end += 1
        y__end += class_size + 1
        # print(class_y)
        # print(class_y_)
        bce = torch.nn.BCEWithLogitsLoss(reduction="none")
        class_loss = bce(class_y_, class_y)
        losses.append(class_loss)
        total_loss += class_loss.squeeze(1)

    # print(coord_y.shape, corner_y_.shape)
    loc_loss = config["loss"]["loss_ratio"] * weighted_loc_loss(coord_y, corner_y_, weights=8)
    # print(total_loss.shape)
    # print(loc_loss.shape)
    total_loss += loc_loss * config["loss"]["class_loss_ratio"]
    losses.append(loc_loss * config["loss"]["class_loss_ratio"])

    if coord_size > 8:
        total_lineLoss = 0
        for index in range(4):
            line = coord_y_[:, index * 2 : index * 2 + 2]
            for coord_index in range(size_per_line):
                line = torch.cat(
                    [
                        line,
                        coord_y_[:, 8 + coord_index * 8 + index * 2 : 8 + coord_index * 8 + index * 2 + 2],
                    ],
                    axis=1,
                )
                # liner = tf.concat([liner,coord_y[:,8+coord_index*8+index*2:8+coord_index*8+index*2+2]],axis=1)
            line = torch.cat([line, coord_y_[:, (index * 2 + 2) % 8 : (index * 2 + 2 + 2) % 8]], axis=1)
            total_lineLoss += line_loss(line)
            # print(line)
            # print(total_lineLoss)
        if use_line_loss:
            losses.append(total_lineLoss * config["loss"]["slop_loss_ratio"])
            # losses.append(total_diff_loss * config["loss"]["diff_loss_ratio"])
            total_loss += total_lineLoss * config["loss"]["slop_loss_ratio"]
            # total_loss += total_diff_loss * config["loss"]["diff_loss_ratio"]
        # else:
        # losses.append(total_slop_loss - total_slop_loss)  # total_slop_loss * slop_loss_ratio)
        # losses.append(total_diff_loss - total_diff_loss)  # total_diff_loss * diff_loss_ratio)
    return total_loss.mean(), losses, [coord_y_, class_y_]


def train(ops, partially_trained = False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = ops["batch_size"]
    epochs = ops["epochs"]
    #img_path = ops["img_folder_path"]
    annotations_path = ops["annotations_folder_path"] + '/*.gt.xml'
    class_size = ops["class_list"]
    Transform = transforms.Compose([])
    cnt = 0
    for annotations in glob.glob(annotations_path):
        annotations = annotations.split('/')[-1]
        img_path = annotations.split('.')[0]
        if os.path.exists(img_path):
            if cnt == 0 :
                dataset = DocData(img_dir=img_path, annotations_path= os.path.join('/home/web_slinger/Downloads/testDataset/background01',annotations), transforms=Transform)
                cnt = cnt + 1
            else :
                temp_dataset = DocData(img_dir=img_path, annotations_path= os.path.join('/home/web_slinger/Downloads/testDataset/background01',annotations), transforms=Transform)
                dataset = torch.utils.data.ConcatDataset([dataset, temp_dataset])
    train_loader = DataLoader(dataset, batch_size=batch_size)
    if partially_trained :
        LDR = torch.load("./model.pth").to(device)
    else :
        LDR = LDRNet(points_size=ops["points_size"]).to(device)
    optimizer = Adam(LDR.parameters(), lr=0.005)

    scheduler = lr_scheduler.MultiStepLR(optimizer, ops["optimizer"]["bounds"])  # gamma=ops["gamma"])

    for epoch in range(epochs):
        step = 0
        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)
            # print(x.shape)
            step += 1
            optimizer.zero_grad()
            loss, debug, _ = Loss(ops, LDR, x, y, True, ops["points_size"] * 2, class_list=ops["class_list"])
            loss.backward()
            optimizer.step()
            if not (step % 10):
                print(f"Current Epoch:{epoch+1}, Current Step:{step}, Current Loss: {loss.item()}")
                # print(debug[0])
        # if not (epoch+1 % 5):
        torch.save(LDR, "./model.pth")
        scheduler.step()
    return LDR


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="/home/web_slinger/Documents/CVI/LDR_NET-main/config.yml", type=str)
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config_file, "r"))
    # print(config)
    net = train(config, partially_trained = True)
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # img = cv2.imread("./datasheet001/frame1.png")
    # img = img.astype(np.float32)
    # img2 = torch.permute(torch.from_numpy((img-127.5) / 255.0), (2, 0, 1)).unsqueeze(0).to(device)
    # result = net(img2)
    # print(result[0])
