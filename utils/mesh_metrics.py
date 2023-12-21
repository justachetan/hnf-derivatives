# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Lint as: python3
# Adapted from https://github.com/google/ldif/blob/master/ldif/inference/metrics.py
"""Computes metrics given predicted and ground truth shape."""
import numpy as np
import scipy


OCCNET_FSCORE_EPS = 1e-09


def sample_points_and_face_normals(mesh, sample_count):
    points, indices = mesh.sample(sample_count, return_index=True)
    points = points.astype(np.float32)
    normals = mesh.face_normals[indices]
    return points, normals


def pointcloud_neighbor_distances_indices(
        source_points, target_points, batch_size=10000, n_jobs=2):
    target_kdtree = scipy.spatial.cKDTree(target_points)
    all_dist, all_idxs = [], []
    for i in range(0, source_points.shape[0], batch_size):
        j = min(source_points.shape[0], i + batch_size)
        points_ij = source_points[i:j]
        distances_ij, indices_ij = target_kdtree.query(points_ij, n_jobs=-1)
        all_dist.append(distances_ij)
        all_idxs.append(indices_ij)
    distances = np.concatenate(all_dist, axis=0)
    indices = np.concatenate(all_idxs, axis=0)
    return distances, indices


def dot_product(a, b):
    if len(a.shape) != 2:
        raise ValueError('Dot Product with input shape: %s' % repr(a.shape))
    if len(b.shape) != 2:
        raise ValueError('Dot Product with input shape: %s' % repr(b.shape))
    return np.sum(a * b, axis=1)


def point_iou(pred_is_inside, gt_is_inside):
    intersection = np.logical_and(pred_is_inside, gt_is_inside).astype(np.float32)
    union = np.logical_or(pred_is_inside, gt_is_inside).astype(np.float32)
    iou = 100.0 * np.sum(intersection) / (np.sum(union) + 1e-05)
    return iou


def percent_below(dists, thresh):
    return np.mean((dists ** 2 <= thresh).astype(np.float32)) * 100.0


def f_score(a_to_b, b_to_a, thresh):
    precision = percent_below(a_to_b, thresh)
    recall = percent_below(b_to_a, thresh)

    return (2 * precision * recall) / (precision + recall + OCCNET_FSCORE_EPS)


def fscore(mesh1,
           mesh2,
           sample_count=100000,
           tau=1e-04,
           points1=None,
           points2=None):
    """Computes the F-Score at tau between two meshes."""
    if points1 is None or points2 is None:
        points1, points2 = get_points(
            mesh1, mesh2, points1, points2, sample_count)
    dist12, _ = pointcloud_neighbor_distances_indices(points1, points2)
    dist21, _ = pointcloud_neighbor_distances_indices(points2, points1)
    f_score_tau = f_score(dist12, dist21, tau)
    return f_score_tau


def mesh_chamfer_via_points(mesh1,
                            mesh2,
                            sample_count=300000,
                            points1=None,
                            points2=None,
                            multiplier=1):
    if points1 is None or points2 is None:
        points1, points2 = get_points(
            mesh1, mesh2, points1, points2, sample_count)
    dist12, _ = pointcloud_neighbor_distances_indices(points1, points2)
    dist21, _ = pointcloud_neighbor_distances_indices(points2, points1)
    chamfer = float(multiplier) * (np.mean(dist12 ** 2) + np.mean(dist21 ** 2))
    return chamfer


def get_points(mesh1, mesh2, points1, points2, sample_count):
    if points1 is not None or points2 is not None:
        assert points1 is not None and points2 is not None
    else:
        points1, _ = sample_points_and_face_normals(mesh1, sample_count)
        points2, _ = sample_points_and_face_normals(mesh2, sample_count)
    return points1, points2


def normal_consistency_with_points(points1, points2, normals1, normals2):
    """Computes the normal consistency metric between two meshes."""
    _, indices12 = pointcloud_neighbor_distances_indices(points1, points2)
    _, indices21 = pointcloud_neighbor_distances_indices(points2, points1)

    normals12 = normals2[indices12]
    normals21 = normals1[indices21]

    # We take abs because the OccNet code takes abs...
    nc12 = np.abs(dot_product(normals1, normals12))
    nc21 = np.abs(dot_product(normals2, normals21))
    nc = 0.5 * np.mean(nc12) + 0.5 * np.mean(nc21)
    return nc



def normal_consistency(mesh1, mesh2, sample_count=300000, return_points=False):
    """Computes the normal consistency metric between two meshes."""
    points1, normals1 = sample_points_and_face_normals(mesh1, sample_count)
    points2, normals2 = sample_points_and_face_normals(mesh2, sample_count)

    nc = normal_consistency_with_points(points1, points2, normals1, normals2)
    if return_points:
        return nc, points1, points2
    return nc


def compute_all_mesh_metrics(
        mesh1, mesh2, sample_count=300000, fs_eps=1e-5):
    nc, points1, points2 = normal_consistency(
        mesh1, mesh2, sample_count, return_points=True)
    fs_tau = fscore(mesh1, mesh2, sample_count, fs_eps, points1, points2)
    fs_2tau = fscore(mesh1, mesh2, sample_count, 2.0 * fs_eps, points1, points2)
    chamfer = mesh_chamfer_via_points(mesh1, mesh2, sample_count, points1,
                                      points2)
    return {
        "normal_consistency": nc,
        "fscore_tau": fs_tau,
        "fscore_2tau": fs_2tau,
        "CD": chamfer
    }


def compute_all_mesh_metrics_with_points(points1, points2, fs_eps=1e-5):
    fs_tau = fscore(None, None, None, fs_eps, points1, points2)
    fs_2tau = fscore(None, None, None, 2.0 * fs_eps, points1, points2)
    chamfer = mesh_chamfer_via_points(None, None, None, points1, points2)
    return {
        "fscore_tau": fs_tau,
        "fscore_2tau": fs_2tau,
        "CD": chamfer
    }


def compute_all_mesh_metrics_with_opcl(
        points1, points2, normals1, normals2, fs_eps=1e-5):
    nc = normal_consistency_with_points(points1, points2, normals1, normals2)
    fs_tau = fscore(None, None, None, fs_eps, points1, points2)
    fs_2tau = fscore(None, None, None, 2.0 * fs_eps, points1, points2)
    chamfer = mesh_chamfer_via_points(None, None, None, points1, points2)
    return {
        "normal_consistency": nc,
        "fscore_tau": fs_tau,
        "fscore_2tau": fs_2tau,
        "CD": chamfer
    }
