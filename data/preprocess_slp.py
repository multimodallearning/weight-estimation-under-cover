import os
import numpy as np
from scipy.ndimage import measurements
from sklearn.cluster import DBSCAN


def depth2cloud(depth, cx, cy, fx, fy, min_=0.01, max_=30):
    height, width = depth.shape
    depth = depth.reshape(-1)
    w, h = np.meshgrid(np.arange(width), np.arange(height))
    w = w.reshape(-1)
    h = h.reshape(-1)
    mask = np.logical_and(depth > min_, depth < max_)

    z = depth[mask]
    x = z * (w[mask] - cx) / fx
    y = z * (h[mask] - cy) / fy

    cloud = np.stack([x, y, z], axis=1)

    return cloud


def ransac(cloud, iterations, inlier_threshold):
    top_inliers = 0
    top_point = None
    top_normal_vector = None
    for it in range(iterations):
        rnd_idx = np.random.randint(low=0, high=cloud.shape[0], size=3)
        sample_points = cloud[rnd_idx, :]
        normal_vector = np.cross(sample_points[0, :] - sample_points[1, :], sample_points[0, :] - sample_points[2, :])
        if np.linalg.norm(normal_vector) == 0.:
            continue
        base_point = sample_points[0, :]
        d = -np.dot(base_point, normal_vector)
        cloud_to_plane_dist = np.abs(np.dot(cloud, normal_vector) + d) / np.linalg.norm(normal_vector)

        inlier_mask = cloud_to_plane_dist < inlier_threshold
        num_inliers = np.sum(inlier_mask)

        if num_inliers > top_inliers:
            top_inliers = num_inliers
            top_point = base_point
            top_normal_vector = normal_vector

    return top_point, top_normal_vector


if __name__ == "__main__":
    root = '../dataset/SLP'

    img_range_dict = {'supine': [1, 16], 'left': [16, 31], 'right': [31, 46]}
    # danaLab and simLab exhibit different heights of the bed, thus use different thresholds for segmenting the bed
    threshold_dict = {'danaLab': 2.2, 'simLab': 2.4}
    ransac_threshold = 0.02

    for cover_cond in ['uncover', 'cover1', 'cover2']:
        print(cover_cond)
        for position in ['supine', 'left', 'right']:
            print(position)
            target_directory = os.path.join(root, '3d_data_{}_{}'.format(position, cover_cond))
            if not os.path.isdir(target_directory):
                os.makedirs(target_directory)
            img_range = img_range_dict[position]
            for lab in ['danaLab', 'simLab']:
                print(lab)
                height_th = threshold_dict[lab]
                subjects = sorted(os.listdir(os.path.join(root, lab)))[:2]

                for subject in subjects:
                    print(subject)
                    subject_path = os.path.join(root, lab, subject)
                    if not os.path.isdir(subject_path):
                        continue

                    for frame_no in range(img_range[0], img_range[1]):

                        depth_path = os.path.join(root, lab, subject, 'depthRaw', cover_cond,
                                                  '{:06d}'.format(frame_no) + '.npy')

                        # load depth image, transform scale to metres and clip to 3m
                        try:
                            depth = np.load(depth_path)
                        except:
                            # for simLab & cover 2, depth maps of subjects 3,4 are not available
                            print('{} is not available'.format(depth_path))
                            continue
                        depth = np.clip(depth, 0, 3000) / 1000

                        # crop the part of the depth map, containing the bed
                        mask = (depth < height_th) * (depth > 0.2)
                        clustered_mask, num_clusters = measurements.label(mask)
                        max_cluster = np.argmax([np.sum(clustered_mask == id) for id in range(1, num_clusters + 1)]) + 1
                        masked_bed = clustered_mask == max_cluster
                        depth[~masked_bed] = 0

                        # convert cropped depth to cloud
                        cloud = depth2cloud(depth, cx=208.1, cy=259.7, fx=367.8, fy=367.8)

                        # save bed cloud
                        cloud_path = lab + '_' + subject + '_' + cover_cond + '_' + '{:03d}'.format(frame_no) + '_bed_pcd.npy'

                        cloud_path = os.path.join(target_directory, cloud_path)
                        np.save(cloud_path, cloud)

                        # for uncovered patients, segment the patient from bed by means of RANSAC and DBSCAN algorithms
                        # determine bed planed through ransac
                        if cover_cond == 'uncover':
                            base_point, normal_vector = ransac(cloud, 1000, 0.01)

                            # extract patient volume as those pixels above the bed plane
                            cloud_to_plane_dist = (np.dot(cloud, normal_vector) - np.dot(base_point,
                                                                                         normal_vector)) / np.linalg.norm(
                                normal_vector)
                            ref_point = np.array([0., 0., 0.])
                            sign = np.sign(
                                (np.dot(ref_point, normal_vector) - np.dot(base_point, normal_vector)) / np.linalg.norm(
                                    normal_vector))
                            if sign > 0:
                                inlier_mask = cloud_to_plane_dist > ransac_threshold
                            else:
                                inlier_mask = cloud_to_plane_dist < -ransac_threshold

                            pat_cloud = cloud[inlier_mask, :]

                            ### remove remaining points that do no belong to patient by clustering the cloud using DBSCAN
                            clustering = DBSCAN(eps=0.025, min_samples=5)
                            cluster_labels = clustering.fit_predict(pat_cloud)
                            most_points = 0
                            corr_label = -1
                            for l in np.unique(cluster_labels):
                                num_points = np.sum(cluster_labels == l)
                                if num_points > most_points:
                                    most_points = num_points
                                    corr_label = l
                            pat_cloud = pat_cloud[cluster_labels == corr_label, :]
                            inlier_mask[inlier_mask] = inlier_mask[inlier_mask] * (cluster_labels == corr_label)

                            # save segmentation mask
                            inlier_mask = inlier_mask.astype(int)
                            label_path = cloud_path.replace('bed_pcd', 'segm_labels')
                            np.save(label_path, inlier_mask)
