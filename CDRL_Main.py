import time
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
from Fuction_Library import *
from Data_Loading import CD_Dataset, Trans_Dataset
from Model_Establishment import Trans_Net, CD_Net


# Default parameter
# For the cross-domain image translation network(CITNet)
batch_size = 32
patch_size = 64
lr = 0.0001
max_epoch = 10
iter_num = 2
isTrain = True
isTest = False

# For the CD network(CDNet)
cd_batch_size = 256
cd_patch_size = 7  # 7 for Shuguang/Texas/Mexico; 11 for France; 3 for Guizhou
cd_lr = 0.001
cd_max_epoch = 18

# Start time
TIME = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
start_time = time.time()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Part 0: Load multimodal images
DATA_PATH = 'Data/Shuguang'
SAVE_PATH = create_folder(DATA_PATH)
im1, im2, im3 = get_data(DATA_PATH)

imgH, imgW = im3.shape
channel_a = im1.shape[-1]
channel_b = im2.shape[-1]

# ########################################################
# # Part 1: Cross-Domain Image Translation
# ########################################################

# Step 1-1: Mask initialization and test dataset generation
mask = np.random.randint(0, 2, size=(imgH, imgW))
test_dataset = Trans_Dataset(im1, im2, mask, patch_size, isTest)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# Step 1-2: Trans_Net initialization and Optimizer setting
CITNet = Trans_Net(channel_a, channel_b).to(device)
optimizer = torch.optim.Adam(CITNet.parameters_list, lr=lr, betas=(0.5, 0.9), weight_decay=5e-4)

# Step 1-3: Model training
print('--------------------CITNet training-----------------------')
for k in range(1, iter_num + 1):

    # Generate train dataset for Trans_Net
    train_dataset = Trans_Dataset(im1, im2, mask, patch_size, isTrain)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print("\n-------------------第{}轮迭代----------------------".format(k))

    for epoch in range(1, max_epoch + 1):
        CITNet.train()
        train_loss = 0
        epoch_start_time = time.time()

        for i, data in enumerate(train_dataloader):
            x_a, x_b, y = data[0], data[1], data[2]
            x_a, x_b, y = x_a.to(device), x_b.to(device), y.to(device)

            optimizer.zero_grad()
            batch_loss = CITNet.update(x_a, x_b, y)

            batch_loss.backward()
            optimizer.step()

            train_loss += batch_loss.item()

        print(
            "End of epoch %d / %d \t Training loss is: %.4f \t Time Taken: %.4f s"
            % (epoch, max_epoch, train_loss / len(train_dataloader), time.time() - epoch_start_time)
        )

# Step 1-4: Model test
    with torch.no_grad():
        CITNet.eval()

        for i, data in enumerate(test_dataloader):
            x_a_t, x_b_t = data[0], data[1]
            x_a, x_b = x_a_t.to(device), x_b_t.to(device)
            data_dict = CITNet(x_a, x_b)

            x_ab = data_dict['x_ab']
            x_ba = data_dict['x_ba']

            c_a = data_dict['c_a']
            c_b = data_dict['c_b']

            c_a_recon = data_dict['c_a_recon']
            c_b_recon = data_dict['c_b_recon']

            # cross feature comparison
            f_a = torch.concat((c_a, c_a_recon), dim=1)
            f_b = torch.concat((c_b_recon, c_b), dim=1)

            DI_crossfeature = torch.sqrt(torch.sum(torch.square(f_a - f_b), dim=1))
            DI_crossfeature = normalization(DI_crossfeature)
            DI_crossfeature = DI_crossfeature.squeeze(0).cpu().numpy()

            _DI_crossfeature = torch.from_numpy(DI_crossfeature)
            mask = change_map(_DI_crossfeature)

            # Performance evaluation for CITNet
            # im_gt = normalization(im3)
            # _, message = calculate_metric(im_gt, mask, "CM")

# ########################################################
# # Part 2: Reliable Sample Generation
# ########################################################

filter_DI_crossfeature = torch.from_numpy(gaussian_filter(DI_crossfeature, 1.5))
mask = change_map(filter_DI_crossfeature)
pre_map = sample_selection(mask, cd_patch_size)

# ########################################################
# # Part 3: Identification of Changes
# ########################################################

# Step 3-1: Generate test dataset for CDNet
CD_test_dataset = CD_Dataset(im1, im2, x_ab, x_ba, pre_map, cd_patch_size, isTest)
CD_test_dataloader = DataLoader(CD_test_dataset, batch_size=cd_batch_size)

# Step 3-2: CDNet and optimizer initialization
CDNet = CD_Net(cd_patch_size*cd_patch_size*16*2, CITNet.con_encoder_params).to(device)
optimizer = torch.optim.Adam(CDNet.parameters_list_classifier, lr=cd_lr, betas=(0.5, 0.9), weight_decay=5e-4)

# Step 3-3: CDNet training
print('\n--------------------CDNet training-----------------------')
for epoch in range(1, cd_max_epoch + 1):

    # Generate train dataset for CDNet
    CD_train_dataset = CD_Dataset(im1, im2, x_ab, x_ba, pre_map, cd_patch_size, isTrain)
    CD_train_dataloader = DataLoader(CD_train_dataset, batch_size=cd_batch_size)

    CDNet.train()
    train_loss = 0
    epoch_start_time = time.time()

    for i, data in enumerate(CD_train_dataloader):
        x_a, x_b, x_ab_t, x_ba_t, y = data[0], data[1], data[2], data[3], data[4]
        x_a, x_b, y = x_a.to(device), x_b.to(device), y.to(device)
        x_ab_t, x_ba_t = x_ab_t.to(device), x_ba_t.to(device)
        y = y.long()

        optimizer.zero_grad()
        loss, train_pred = CDNet.update(CITNet.gen_a.enc_content, CITNet.gen_b.enc_content, x_a, x_b, x_ab_t, x_ba_t, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    tmp_loss = train_loss/len(CD_train_dataloader)

    print(
        "End of epoch %d / %d \t Training loss is: %.4f \t Time Taken: %.4f s"
        % (epoch, cd_max_epoch, tmp_loss, time.time() - epoch_start_time)
    )

# Step 3-4: Change detection
print('\n--------------------Change Detection-----------------------')
with torch.no_grad():
    CDNet.eval()
    pred_label = []

    for i, data in enumerate(CD_test_dataloader):
        x_a, x_b, x_ab, x_ba, y = data[0], data[1], data[2], data[3], data[4]
        x_a, x_b = x_a.to(device), x_b.to(device)
        x_ab, x_ba = x_ab.to(device), x_ba.to(device)
        y = y.long()

        _, test_pred = CDNet(CITNet.gen_a.enc_content, CITNet.gen_b.enc_content, x_a, x_b, x_ab, x_ba)

        pred_label.append(test_pred)

    pred_label = torch.cat(pred_label).cpu().numpy()

# ########################################################
# # Part 4: Performance evaluation
# ########################################################

map_label = np.zeros((imgH * imgW))
pre_map_label = pre_map.flatten()

map_label[pre_map_label == 0] = 0
map_label[pre_map_label == 1] = 1
map_label[pre_map_label == 2] = pred_label.flatten()
im_gt = normalization(im3)

map = (map_label.reshape(imgH, imgW)).astype(np.uint8)
_, message = calculate_metric(im_gt, map, "CM")
write_images(map, SAVE_PATH, 'CM', 'map')

# Total time
end_time = time.time()
total_time = end_time - start_time
print('\n Total cost time is : {:.6f}s'.format(total_time))

# ########################################################
# # Part 5: Results saving
# ########################################################

with open(Path(SAVE_PATH, "CM_evaluate.txt"), "a") as f:
    f.write(message + "\n")
    f.close()

write_images(DI_crossfeature, SAVE_PATH, 'DI_crossfeature', 'map')
write_images(x_a_t, SAVE_PATH, 'x_a', 'img')
write_images(x_b_t, SAVE_PATH, 'x_b', 'img')
write_images(data_dict['x_ab'], SAVE_PATH, 'x_ab', 'img')
write_images(data_dict['x_ba'], SAVE_PATH, 'x_ba', 'img')
write_images(data_dict['x_a_recon'], SAVE_PATH, 'x_a_recon', 'img')
write_images(data_dict['x_b_recon'], SAVE_PATH, 'x_b_recon', 'img')
write_images(cam(1 - DI_crossfeature), SAVE_PATH, 'DI_crossfeature_cam', 'map')

conf_map = np.zeros_like(map)
conf_map = np.tile(conf_map[..., np.newaxis], (1, 1, 3))
conf_map[np.logical_and(im3, map)] = [1, 1, 1]
conf_map[np.logical_and(im3, np.logical_not(map)), :] = [0, 1, 0]
conf_map[np.logical_and(np.logical_not(im3), map), :] = [1, 0, 0]
write_images(conf_map, SAVE_PATH, 'CM_Confusion', 'map')
