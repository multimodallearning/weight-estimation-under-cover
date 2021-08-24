import numpy as np
import glob
import torch.utils.data


class SLPUncoverDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, phase):
        position = cfg.SLP_DATASET.POSITION
        cover_cond = cfg.SLP_DATASET.COVER_CONDITION
        val_split = cfg.SLP_DATASET.VAL_SPLIT

        if position == 'all':
            all_positions = ['supine', 'left', 'right']
        elif position == 'lateral':
            all_positions = ['left', 'right']
        else:
            all_positions = [position]

        self.input_list = []
        self.target_list = []

        for position in all_positions:
            if cover_cond == 'cover12':
                input_list_1 = sorted(glob.glob('../dataset/SLP/3d_data_{}_cover1/*bed_pcd.npy'.format(position)))
                input_list_2 = sorted(glob.glob('../dataset/SLP/3d_data_{}_cover2/*bed_pcd.npy'.format(position)))
            elif cover_cond in ['cover1', 'cover2']:
                input_list = sorted(glob.glob('../dataset/SLP/3d_data_{}_{}/*bed_pcd.npy'.format(position, cover_cond)))
            else:
                raise ValueError

            target_list = sorted(glob.glob('../dataset/SLP/3d_data_{}_uncover/*bed_pcd.npy'.format(position)))

            if cover_cond == 'cover12':
                if phase == 'train':
                    input_list = input_list_1[:900] + input_list_2[:900]
                    target_list = target_list[:900] + target_list[:900]
                else:
                    if val_split == 'dana':
                        input_list = input_list_1[900:1530] + input_list_2[900:1530]
                        target_list = target_list[900:1530] + target_list[900:1530]
                    elif val_split == 'sim':
                        input_list = input_list_1[1530:] + input_list_2[1530:]
                        # for cover2 of simLab, depth data is not available for subjects 3 and 4
                        target_list = target_list[1530:] + target_list[1530:1560] + target_list[1590:]
                    else:
                        raise ValueError
            else:
                if phase == 'train':
                    input_list = input_list[:900]
                    target_list = target_list[:900]
                else:
                    if val_split == 'dana':
                        input_list = input_list[900:1530]
                        target_list = target_list[900:1530]
                    elif val_split == 'sim':
                        input_list = input_list[1530:]
                        if cover_cond == 'cover1':
                            target_list = target_list[1530:]
                        elif cover_cond == 'cover2':
                            target_list[1530:1560] + target_list[1590:]
                    else:
                        raise ValueError

            self.input_list.extend(input_list)
            self.target_list.extend(target_list)

        self.is_train = True if phase == 'train' else False
        self.use_patient_segmentation = cfg.SLP_DATASET.USE_PATIENT_SEGMENTATION
        self.position = cfg.SLP_DATASET.POSITION
        self.rot_degree = cfg.INPUT.ROTATION_DEGREE

        self.grid_shape = np.array(cfg.INPUT.VOXEL_GRID_SHAPE)
        self.grid_size = np.array(cfg.INPUT.VOXEL_GRID_SIZE)
        self.min_cloud_values = np.array(cfg.INPUT.MIN_CLOUD_VALUES)

    def __getitem__(self, idx):
        input_pcd = np.float32(np.load(self.input_list[idx]))
        target_pcd = np.float32(np.load(self.target_list[idx]))
        if self.use_patient_segmentation:
            target_path = self.target_list[idx]
            segm_path = target_path.replace('bed_pcd', 'segm_labels')
            segm = np.load(segm_path)
            target_pcd = target_pcd[segm == 1, :]

        mean = np.mean(input_pcd, axis=0)
        input_pcd -= mean
        target_pcd -= mean

        # rotation around the z-axis
        if self.is_train and self.rot_degree > 0.:
            theta = np.deg2rad(np.random.uniform(-self.rot_degree, self.rot_degree))
            rotation_matrix = np.float32(np.array([[np.cos(theta), -np.sin(theta), 0],
                                                   [np.sin(theta), np.cos(theta), 0],
                                                   [0., 0., 1.], ]))
            input_pcd = np.dot(input_pcd, rotation_matrix)
            target_pcd = np.dot(target_pcd, rotation_matrix)

        input_vol = self.cloud2vol(input_pcd)
        target_vol = self.cloud2vol(target_pcd)
        target_vol = np.int64(target_vol[0, :, :, :])

        return input_vol, target_vol, idx

    def __len__(self):
        return len(self.input_list)

    def cloud2vol(self, cloud):
        cloud -= self.min_cloud_values
        cloud = cloud / (self.grid_size / self.grid_shape)
        cloud = np.floor(cloud).astype(int)
        for i in range(3):
            cloud[:, i] = np.clip(cloud[:, i], 0, self.grid_shape[i] - 1)
        vol = np.zeros([1, *self.grid_shape])
        vol[0, cloud[:, 0], cloud[:, 1], cloud[:, 2]] = 1.
        vol = np.float32(vol)

        return vol
