import numpy as np
import glob
import torch.utils.data


class SLPWeightDataset(torch.utils.data.Dataset):
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

        self.data_list = []
        self.physics = []

        danalab_physics = np.load('../dataset/SLP/danaLab/physiqueData.npy')
        simlab_physics = np.load('../dataset/SLP/simLab/physiqueData.npy')

        for position in all_positions:

            if cover_cond == 'cover12':
                cloud_list_1 = sorted(
                    glob.glob('../dataset/SLP/3d_data_{}_cover1/*bed_pcd.npy'.format(position)))
                cloud_list_2 = sorted(
                    glob.glob('../dataset/SLP/3d_data_{}_cover2/*bed_pcd.npy'.format(position)))
            elif cover_cond in ['uncover', 'cover1', 'cover2']:
                #TODO: in pre-processing script, use correct naming for destination folders (3d_data_xxx_UNCOVER)
                cloud_list = sorted(glob.glob('../dataset/SLP/3d_data_{}_{}/*bed_pcd.npy'.format(position, cover_cond)))
            else:
                raise ValueError

            if cover_cond == 'cover12':
                if phase == 'train':
                    # use first 60 subjects for training
                    data_list = cloud_list_1[:900] + cloud_list_2[:900]
                    physics = np.concatenate((danalab_physics[:60, :], danalab_physics[:60, :]), axis=0)
                elif phase == 'val':
                    if val_split == 'dana':
                        # use subjects 61-102 for evaluation on danaLab data
                        data_list = cloud_list_1[900:1530] + cloud_list_2[900:1530]
                        physics = np.concatenate((danalab_physics[60:, :], danalab_physics[60:, :]), axis=0)
                    elif val_split == 'sim':
                        data_list = cloud_list_1[1530:] + cloud_list_2[1530:]
                        # for cover2 of simLab, depth data is not available for subjects 3 and 4
                        physics = np.concatenate(
                            (simlab_physics, simlab_physics[:2, :], simlab_physics[4:, :]), axis=0)
                    else:
                        raise ValueError
            else:
                if phase == 'train':
                    data_list = cloud_list[:900]
                    physics = danalab_physics[:60, :]
                else:
                    if val_split == 'dana':
                        data_list = cloud_list[900:1530]
                        physics = danalab_physics[60:, :]
                    elif val_split == 'sim':
                        data_list = cloud_list[1530:]
                        if cover_cond == 'cover2':
                            # for cover2 of simLab, depth data is not available for subjects 3 and 4
                            physics = np.concatenate((simlab_physics[:2, :], simlab_physics[4:, :]), axis=0)
                        else:
                            physics = simlab_physics
                    else:
                        raise ValueError

            self.data_list.extend(data_list)
            if len(self.physics) == 0:
                self.physics = physics
            else:
                self.physics = np.concatenate((self.physics, physics), axis=0)

        self.num_points = cfg.INPUT.NUM_POINTS
        self.normalize_output = cfg.INPUT.NORMALIZE_OUTPUT
        self.min_weight_train = np.min(danalab_physics[:60, 2])
        self.max_weight_train = np.max(danalab_physics[:60, 2])

        self.is_train = True if phase == 'train' else False
        self.rot_degree = cfg.INPUT.ROTATION_DEGREE

        self.voxelize = cfg.INPUT.VOXELIZE
        self.grid_shape = np.array(cfg.INPUT.VOXEL_GRID_SHAPE)
        self.grid_size = np.array(cfg.INPUT.VOXEL_GRID_SIZE)
        self.min_cloud_values = np.array(cfg.INPUT.MIN_CLOUD_VALUES)

    def __getitem__(self, idx):
        pcd = np.float32(np.load(self.data_list[idx]))
        pcd -= np.mean(pcd, axis=0)

        # rotatio around the z-axis
        if self.is_train and self.rot_degree > 0.:
            theta = np.deg2rad(np.random.uniform(-self.rot_degree, self.rot_degree))
            rotation_matrix = np.float32(np.array([[np.cos(theta), -np.sin(theta), 0],
                                                   [np.sin(theta), np.cos(theta), 0],
                                                   [0., 0., 1.], ]))
            pcd = np.dot(pcd, rotation_matrix)

        if not self.voxelize:
            pcd = torch.from_numpy(np.transpose(pcd))
            # subsample input points
            if self.num_points > 0:
                pts_size = pcd.size(1)

                if pts_size >= self.num_points:
                    if self.is_train:
                        permutation = torch.randperm(pcd.size(1))
                    else:
                        permutation = torch.from_numpy(np.random.default_rng(12345).permutation(pcd.size(1)))
                    pcd = pcd[:, permutation]
                    pcd = pcd[:, :self.num_points]
                else:
                    if self.is_train:
                        pts_idx = torch.from_numpy(np.random.choice(pts_size, self.num_points, replace=True))
                    else:
                        pts_idx = torch.from_numpy(
                            np.random.default_rng(12345).choice(pts_size, self.num_points, replace=True))
                    pcd = pcd[:, pts_idx]

            input = pcd

        else:
            input = self.cloud2vol(pcd)

        weight = self.physics[int(idx / 15), 2]
        # normalize target weight to [0, 1]
        if self.normalize_output:
            weight = (weight - self.min_weight_train) / (self.max_weight_train - self.min_weight_train)

        target = torch.from_numpy(np.float32([weight]))

        return input, target, idx

    def __len__(self):
        return len(self.data_list)

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
