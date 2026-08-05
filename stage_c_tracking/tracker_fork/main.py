"""Vendored from eddyhkchiu/mahalanobis_3d_multi_object_tracking (main.py).

Compatibility fixes applied:
  - sklearn.utils.linear_assignment_ (removed in sklearn 1.3) replaced with
    scipy.optimize.linear_sum_assignment (returns row_ind, col_ind separately).
  - Hardcoded /juno/u/hkchiu/... paths replaced with configurable args.
  - glob2 dependency in utils.py removed (use os.walk).

Stage C modification (the SINGLE change):
  - KalmanBoxTracker stores self.R_base = self.kf.R.copy() at birth.
  - set_reliability(s_t, kappa) rescales self.kf.R = self.R_base * (1 + kappa*(1-s_t))
    per frame before kf.update(). S_t is per-frame (looked up by sample_token),
    applied to every active track's R before that frame's update.
  - When --score-file is None or identity_scores.npz, R(t) = R_base exactly (H1).

Nothing else is changed: same association metric (Mahalanobis with
trks_S = H·P·Hᵀ + R), same track birth/death, same motion model
(CV with state [x,y,z,θ,l,w,h,ẋ,ẏ,ż,θ̇]).
"""

from __future__ import print_function
import os.path, copy, numpy as np, time, sys
from numba import jit
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from utils import load_list_from_folder, fileparts, mkdir_if_missing
from scipy.spatial import ConvexHull
from covariance import Covariance
import json
import argparse
from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.tracking.data_classes import TrackingBox
from nuscenes.eval.detection.data_classes import DetectionBox
from pyquaternion import Quaternion
from tqdm import tqdm


@jit
def poly_area(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


@jit
def box3d_vol(corners):
    a = np.sqrt(np.sum((corners[0, :] - corners[1, :]) ** 2))
    b = np.sqrt(np.sum((corners[1, :] - corners[2, :]) ** 2))
    c = np.sqrt(np.sum((corners[0, :] - corners[4, :]) ** 2))
    return a * b * c


@jit
def convex_hull_intersection(p1, p2):
    inter_p = polygon_clip(p1, p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0


def polygon_clip(subjectPolygon, clipPolygon):
    def inside(p):
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

    def computeIntersection():
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

    outputList = subjectPolygon
    cp1 = clipPolygon[-1]
    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]
        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s):
                    outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())
            s = e
        cp1 = cp2
        if len(outputList) == 0:
            return None
    return outputList


def iou3d(corners1, corners2):
    rect1 = [(corners1[i, 0], corners1[i, 2]) for i in range(3, -1, -1)]
    rect2 = [(corners2[i, 0], corners2[i, 2]) for i in range(3, -1, -1)]
    area1 = poly_area(np.array(rect1)[:, 0], np.array(rect1)[:, 1])
    area2 = poly_area(np.array(rect2)[:, 0], np.array(rect2)[:, 1])
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area / (area1 + area2 - inter_area)
    ymax = min(corners1[0, 1], corners2[0, 1])
    ymin = max(corners1[4, 1], corners2[4, 1])
    inter_vol = inter_area * max(0.0, ymax - ymin)
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d


@jit
def roty(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


@jit
def rotz(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


def convert_3dbox_to_8corner(bbox3d_input, nuscenes_to_kitti=False):
    bbox3d = copy.copy(bbox3d_input)
    if nuscenes_to_kitti:
        bbox3d_nuscenes = copy.copy(bbox3d)
        bbox3d[0] = bbox3d_nuscenes[1]
        bbox3d[1] = -bbox3d_nuscenes[2]
        bbox3d[2] = -bbox3d_nuscenes[0]
        bbox3d[3] = -bbox3d_nuscenes[3]
        bbox3d[4] = bbox3d_nuscenes[5]
        bbox3d[5] = bbox3d_nuscenes[4]

    R = roty(bbox3d[3])
    l = bbox3d[4]
    w = bbox3d[5]
    h = bbox3d[6]
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + bbox3d[0]
    corners_3d[1, :] = corners_3d[1, :] + bbox3d[1]
    corners_3d[2, :] = corners_3d[2, :] + bbox3d[2]
    return np.transpose(corners_3d)


class KalmanBoxTracker(object):
  """
  This class represents the internel state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self, bbox3D, info, covariance_id=0, track_score=None, tracking_name='car', use_angular_velocity=False):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    if not use_angular_velocity:
      self.kf = KalmanFilter(dim_x=10, dim_z=7)
      self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0],      # state transition matrix
                            [0,1,0,0,0,0,0,0,1,0],
                            [0,0,1,0,0,0,0,0,0,1],
                            [0,0,0,1,0,0,0,0,0,0],
                            [0,0,0,0,1,0,0,0,0,0],
                            [0,0,0,0,0,1,0,0,0,0],
                            [0,0,0,0,0,0,1,0,0,0],
                            [0,0,0,0,0,0,0,1,0,0],
                            [0,0,0,0,0,0,0,0,1,0],
                            [0,0,0,0,0,0,0,0,0,1]])

      self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0],      # measurement function,
                            [0,1,0,0,0,0,0,0,0,0],
                            [0,0,1,0,0,0,0,0,0,0],
                            [0,0,0,1,0,0,0,0,0,0],
                            [0,0,0,0,1,0,0,0,0,0],
                            [0,0,0,0,0,1,0,0,0,0],
                            [0,0,0,0,0,0,1,0,0,0]])
    else:
      # with angular velocity
      self.kf = KalmanFilter(dim_x=11, dim_z=7)
      self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0,0],      # state transition matrix
                            [0,1,0,0,0,0,0,0,1,0,0],
                            [0,0,1,0,0,0,0,0,0,1,0],
                            [0,0,0,1,0,0,0,0,0,0,1],
                            [0,0,0,0,1,0,0,0,0,0,0],
                            [0,0,0,0,0,1,0,0,0,0,0],
                            [0,0,0,0,0,0,1,0,0,0,0],
                            [0,0,0,0,0,0,0,1,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0],
                            [0,0,0,0,0,0,0,0,0,1,0],
                            [0,0,0,0,0,0,0,0,0,0,1]])

      self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0,0],      # measurement function,
                            [0,1,0,0,0,0,0,0,0,0,0],
                            [0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,1,0,0,0,0,0,0,0],
                            [0,0,0,0,1,0,0,0,0,0,0],
                            [0,0,0,0,0,1,0,0,0,0,0],
                            [0,0,0,0,0,0,1,0,0,0,0]])

    # Initialize the covariance matrix, see covariance.py for more details
    if covariance_id == 0: # exactly the same as AB3DMOT baseline
      self.kf.P[7:,7:] *= 1000. #state uncertainty, give high uncertainty to the unobservable initial velocities, covariance matrix
      self.kf.P *= 10.
      self.kf.Q[7:,7:] *= 0.01
    elif covariance_id == 1: # for kitti car, not supported
      covariance = Covariance(covariance_id)
      self.kf.P = covariance.P
      self.kf.Q = covariance.Q
      self.kf.R = covariance.R
    elif covariance_id == 2: # for nuscenes
      covariance = Covariance(covariance_id)
      self.kf.P = covariance.P[tracking_name]
      self.kf.Q = covariance.Q[tracking_name]
      self.kf.R = covariance.R[tracking_name]
      if not use_angular_velocity:
        self.kf.P = self.kf.P[:-1,:-1]
        self.kf.Q = self.kf.Q[:-1,:-1]
    else:
      assert(False)

    # --- Stage C R(t) modification: store R_base at track birth ---
    # R_base is the per-class R from covariance.R[tracking_name] (covariance_id==2).
    # set_reliability() rescales self.kf.R from R_base each frame before kf.update().
    self.R_base = self.kf.R.copy()
    assert np.all(np.diag(self.R_base) > 0), "R_base must be positive-definite (positive diagonal)"
    # --- end Stage C modification ---

    self.kf.x[:7] = bbox3D.reshape((7, 1))

    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 1           # number of total hits including the first detection
    self.hit_streak = 1     # number of continuing hit considering the first detection
    self.first_continuing_hit = 1
    self.still_first = True
    self.age = 0
    self.info = info        # other info
    self.track_score = track_score
    self.tracking_name = tracking_name
    self.use_angular_velocity = use_angular_velocity

  # --- Stage C R(t) modification: the single new method ---
  def set_reliability(self, s_t, kappa):
    """Rescale measurement noise: R(t) = R_base * (1 + kappa * (1 - s_t)).

    Called per-frame (before kf.update()) with the current sample_token's S_t.
    S_t is per-frame, not per-track: the same scalar is applied to every active
    track's R before that frame's update. Do NOT special-case scalar == 1.0:
    R_base * 1.0 is exact in IEEE 754, so branching would bypass the arithmetic
    the H1 identity check exists to test.
    """
    scalar = 1.0 + kappa * (1.0 - s_t)
    assert scalar > 0, f"R(t) scalar must be positive, got {scalar}"
    self.kf.R = self.R_base * scalar
  # --- end Stage C modification ---

  def update(self, bbox3D, info):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1          # number of continuing hit
    if self.still_first:
      self.first_continuing_hit += 1      # number of continuing hit in the fist time

    ######################### orientation correction
    if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the range
    if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

    new_theta = bbox3D[3]
    if new_theta >= np.pi: new_theta -= np.pi * 2    # make the theta still in the range
    if new_theta < -np.pi: new_theta += np.pi * 2
    bbox3D[3] = new_theta

    predicted_theta = self.kf.x[3]
    if abs(new_theta - predicted_theta) > np.pi / 2.0 and abs(new_theta - predicted_theta) < np.pi * 3 / 2.0:     # if the angle of two theta is not acute angle
      self.kf.x[3] += np.pi
      if self.kf.x[3] > np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the range
      if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

    # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
    if abs(new_theta - self.kf.x[3]) >= np.pi * 3 / 2.0:
      if new_theta > 0: self.kf.x[3] += np.pi * 2
      else: self.kf.x[3] -= np.pi * 2

    #########################

    self.kf.update(bbox3D)

    if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the range
    if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2
    self.info = info

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    self.kf.predict()
    if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2
    if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
      self.still_first = False
    self.time_since_update += 1
    self.history.append(self.kf.x)
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return self.kf.x[:7].reshape((7, ))


def angle_in_range(angle):
  if angle > np.pi:
    angle -= 2 * np.pi
  if angle < -np.pi:
    angle += 2 * np.pi
  return angle


def diff_orientation_correction(det, trk):
  diff = det - trk
  diff = angle_in_range(diff)
  if diff > np.pi / 2:
    diff -= np.pi
  if diff < -np.pi / 2:
    diff += np.pi
  diff = angle_in_range(diff)
  return diff


def greedy_match(distance_matrix):
  '''
  Find the one-to-one matching using greedy allgorithm choosing small distance
  distance_matrix: (num_detections, num_tracks)
  '''
  matched_indices = []

  num_detections, num_tracks = distance_matrix.shape
  distance_1d = distance_matrix.reshape(-1)
  index_1d = np.argsort(distance_1d)
  index_2d = np.stack([index_1d // num_tracks, index_1d % num_tracks], axis=1)

  detection_id_matches_to_tracking_id = [-1] * num_detections
  tracking_id_matches_to_detection_id = [-1] * num_tracks
  for sort_i in range(index_2d.shape[0]):
    detection_id = int(index_2d[sort_i][0])
    tracking_id = int(index_2d[sort_i][1])
    if tracking_id_matches_to_detection_id[tracking_id] == -1 and detection_id_matches_to_tracking_id[detection_id] == -1:
      tracking_id_matches_to_detection_id[tracking_id] = detection_id
      detection_id_matches_to_tracking_id[detection_id] = tracking_id
      matched_indices.append([detection_id, tracking_id])

  matched_indices = np.array(matched_indices)
  return matched_indices


def _linear_assignment_compat(cost_matrix):
  """scipy.optimize.linear_sum_assignment wrapper returning [[r,c],...] like
  the old sklearn.linear_assignment_ signature."""
  row_ind, col_ind = linear_sum_assignment(cost_matrix)
  if len(row_ind) == 0:
    return np.empty((0, 2), dtype=int)
  return np.stack([row_ind, col_ind], axis=1)


def associate_detections_to_trackers(detections,trackers,iou_threshold=0.1,
  use_mahalanobis=False, dets=None, trks=None, trks_S=None, mahalanobis_threshold=0.1, print_debug=False, match_algorithm='greedy'):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  detections:  N x 8 x 3
  trackers:    M x 8 x 3

  dets: N x 7
  trks: M x 7
  trks_S: N x 7 x 7

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,8,3),dtype=int)
  iou_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)
  distance_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)

  if use_mahalanobis:
    assert(dets is not None)
    assert(trks is not None)
    assert(trks_S is not None)

  if use_mahalanobis and print_debug:
    print('dets.shape: ', dets.shape)
    print('dets: ', dets)
    print('trks.shape: ', trks.shape)
    print('trks: ', trks)
    print('trks_S.shape: ', trks_S.shape)
    print('trks_S: ', trks_S)
    S_inv = [np.linalg.inv(S_tmp) for S_tmp in trks_S]  # 7 x 7
    S_inv_diag = [S_inv_tmp.diagonal() for S_inv_tmp in S_inv]# 7
    print('S_inv_diag: ', S_inv_diag)

  for d,det in enumerate(detections):
    for t,trk in enumerate(trackers):
      if use_mahalanobis:
        S_inv = np.linalg.inv(trks_S[t]) # 7 x 7
        diff = np.expand_dims(dets[d] - trks[t], axis=1) # 7 x 1
        # manual reversed angle by 180 when diff > 90 or < -90 degree
        corrected_angle_diff = diff_orientation_correction(dets[d][3], trks[t][3])
        diff[3] = corrected_angle_diff
        distance_matrix[d, t] = np.sqrt(np.matmul(np.matmul(diff.T, S_inv), diff)[0][0])
      else:
        iou_matrix[d,t] = iou3d(det,trk)[0]             # det: 8 x 3, trk: 8 x 3
        distance_matrix = -iou_matrix

  if match_algorithm == 'greedy':
    matched_indices = greedy_match(distance_matrix)
  elif match_algorithm == 'pre_threshold':
    if use_mahalanobis:
      to_max_mask = distance_matrix > mahalanobis_threshold
      distance_matrix[to_max_mask] = mahalanobis_threshold + 1
    else:
      to_max_mask = iou_matrix < iou_threshold
      distance_matrix[to_max_mask] = 0
      iou_matrix[to_max_mask] = 0
    matched_indices = _linear_assignment_compat(distance_matrix)      # houngarian algorithm
  else:
    matched_indices = _linear_assignment_compat(distance_matrix)      # houngarian algorithm

  if print_debug:
    print('distance_matrix.shape: ', distance_matrix.shape)
    print('distance_matrix: ', distance_matrix)
    print('matched_indices: ', matched_indices)

  unmatched_detections = []
  for d,det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t,trk in enumerate(trackers):
    if len(matched_indices) == 0 or (t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    match = True
    if use_mahalanobis:
      if distance_matrix[m[0],m[1]] > mahalanobis_threshold:
        match = False
    else:
      if(iou_matrix[m[0],m[1]]<iou_threshold):
        match = False
    if not match:
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  if print_debug:
    print('matches: ', matches)
    print('unmatched_detections: ', unmatched_detections)
    print('unmatched_trackers: ', unmatched_trackers)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class AB3DMOT(object):
  def __init__(self,covariance_id=0, max_age=2,min_hits=3, tracking_name='car', use_angular_velocity=False, tracking_nuscenes=False):
    """
    observation:
      before reorder: [h, w, l, x, y, z, rot_y]
      after reorder:  [x, y, z, rot_y, l, w, h]
    state:
      [x, y, z, rot_y, l, w, h, x_dot, y_dot, z_dot]
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.trackers = []
    self.frame_count = 0
    self.reorder = [3, 4, 5, 6, 2, 1, 0]
    self.reorder_back = [6, 5, 4, 0, 1, 2, 3]
    self.covariance_id = covariance_id
    self.tracking_name = tracking_name
    self.use_angular_velocity = use_angular_velocity
    self.tracking_nuscenes = tracking_nuscenes

  def update(self,dets_all, match_distance, match_threshold, match_algorithm, seq_name):
    """
    Params:
      dets_all: dict
        dets - a numpy array of detections in the format [[x,y,z,theta,l,w,h],[x,y,z,theta,l,w,h],...]
        info: a array of other info for each det
    Requires: this method must be called once for each frame even with empty detections.
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    dets, info = dets_all['dets'], dets_all['info']         # dets: N x 7, float numpy array
    dets = dets[:, self.reorder]


    self.frame_count += 1

    print_debug = False
    if False and seq_name == '2f56eb47c64f43df8902d9f88aa8a019' and self.frame_count >= 25 and self.frame_count <= 30:
      print_debug = True
      print('self.frame_count: ', self.frame_count)
    if print_debug:
      for trk_tmp in self.trackers:
        print('trk_tmp.id: ', trk_tmp.id)

    trks = np.zeros((len(self.trackers),7))         # N x 7 , #get predicted locations from existing trackers.
    to_del = []
    ret = []
    for t,trk in enumerate(trks):
      pos = self.trackers[t].predict().reshape((-1,))  # flatten to 1D (numpy 1.24+ compat)
      trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6]]
      if(np.any(np.isnan(pos))):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)

    if print_debug:
      for trk_tmp in self.trackers:
        print('trk_tmp.id: ', trk_tmp.id)

    dets_8corner = [convert_3dbox_to_8corner(det_tmp, match_distance == 'iou' and self.tracking_nuscenes) for det_tmp in dets]
    if len(dets_8corner) > 0: dets_8corner = np.stack(dets_8corner, axis=0)
    else: dets_8corner = []

    trks_8corner = [convert_3dbox_to_8corner(trk_tmp, match_distance == 'iou' and self.tracking_nuscenes) for trk_tmp in trks]
    trks_S = [np.matmul(np.matmul(tracker.kf.H, tracker.kf.P), tracker.kf.H.T) + tracker.kf.R for tracker in self.trackers]

    if len(trks_8corner) > 0:
      trks_8corner = np.stack(trks_8corner, axis=0)
      trks_S = np.stack(trks_S, axis=0)
    if match_distance == 'iou':
      matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets_8corner, trks_8corner, iou_threshold=match_threshold, print_debug=print_debug, match_algorithm=match_algorithm)
    else:
      matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets_8corner, trks_8corner, use_mahalanobis=True, dets=dets, trks=trks, trks_S=trks_S, mahalanobis_threshold=match_threshold, print_debug=print_debug, match_algorithm=match_algorithm)

    #update matched trackers with assigned detections
    for t,trk in enumerate(self.trackers):
      if t not in unmatched_trks:
        d = matched[np.where(matched[:,1]==t)[0],0]     # a list of index
        # --- Stage C R(t) modification: apply set_reliability before kf.update() ---
        # self._current_s_t and self._kappa are set per-frame by the caller
        # (track_nuscenes) before calling update(). When no score file is loaded
        # (baseline), _current_s_t defaults to 1.0 so R(t)=R_base exactly.
        if hasattr(self, '_current_s_t'):
          trk.set_reliability(self._current_s_t, self._kappa)
        # --- end Stage C modification ---
        trk.update(dets[d,:][0], info[d, :][0])
        detection_score = info[d, :][0][-1]
        trk.track_score = detection_score

    #create and initialise new trackers for unmatched detections
    for i in unmatched_dets:        # a scalar of index
        detection_score = info[i][-1]
        track_score = detection_score
        trk = KalmanBoxTracker(dets[i,:], info[i, :], self.covariance_id, track_score, self.tracking_name, use_angular_velocity)
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()      # bbox location
        d = d[self.reorder_back]

        if((trk.time_since_update < self.max_age) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):
          # --- Stage C Phase 6.3: append Kalman-smoothed velocity (vx, vy) ---
          # kf.x[7]=x_dot, kf.x[8]=y_dot in the global frame (matches nuScenes
          # velocity convention). Emitted so the tracking JSON can be converted
          # to detection format for the mAVE velocity-smoothing comparison.
          vx = float(trk.kf.x[7])
          vy = float(trk.kf.x[8])
          ret.append(np.concatenate((d, [trk.id+1], trk.info[:-1], [trk.track_score], [vx, vy])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        #remove dead tracklet
        if(trk.time_since_update >= self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret)      # x, y, z, theta, l, w, h, ID, other info, confidence
    return np.empty((0,15 + 7))


NUSCENES_TRACKING_NAMES = [
  'bicycle',
  'bus',
  'car',
  'motorcycle',
  'pedestrian',
  'trailer',
  'truck'
]

def format_sample_result(sample_token, tracking_name, tracker):
  '''
  Input:
    tracker: (11): [h, w, l, x, y, z, rot_y], tracking_id, tracking_score, vx, vy
      vx, vy are the Kalman-smoothed global-frame velocities (kf.x[7], kf.x[8]),
      emitted for the Phase 6.3 mAVE story (velocity smoothing). The tracking
      eval ignores this field.
  Output:
    sample_result
  '''
  rotation = Quaternion(axis=[0, 0, 1], angle=tracker[6]).elements
  sample_result = {
    'sample_token': sample_token,
    'translation': [tracker[3], tracker[4], tracker[5]],
    'size': [tracker[1], tracker[2], tracker[0]],
    'rotation': [rotation[0], rotation[1], rotation[2], rotation[3]],
    'velocity': [float(tracker[9]), float(tracker[10])],
    'tracking_id': str(int(tracker[7])),
    'tracking_name': tracking_name,
    'tracking_score': tracker[8]
  }

  return sample_result


def load_score_file(score_file):
  """Load a Stage C score .npz and return {sample_token: s_t} dict.

  Returns None if score_file is None (baseline: R(t)=R_base, s_t defaults to 1.0).
  """
  if score_file is None:
    return None
  d = np.load(score_file, allow_pickle=True)
  tokens = d["tokens"].astype(str)
  s_t = d["s_t"].astype(np.float64)
  return {t: float(s) for t, s in zip(tokens, s_t)}


def track_nuscenes(data_split, covariance_id, match_distance, match_threshold,
                  match_algorithm, save_root, use_angular_velocity,
                  detection_file=None, data_root=None, version='v1.0-trainval',
                  output_path=None, score_file=None, kappa=3.0):
  """Stage C entry point.

  Args:
    detection_file: path to results_nusc.json (BEVFusion detections). Required.
    data_root: nuScenes dataroot. Required.
    score_file: path to scores_bev_real/<condition>_scores.npz, or None for
      baseline (R(t)=R_base), or identity_scores.npz (H1: s_t=1.0 => R(t)=R_base).
    kappa: R(t) inflation coefficient. Default 3.0 (pre-sweep midpoint).
  """
  save_dir = os.path.dirname(output_path) if output_path else os.path.join(save_root, data_split)
  mkdir_if_missing(save_dir)
  if output_path is None:
    output_path = os.path.join(save_dir, 'results_val_probabilistic_tracking.json')

  # --- Stage C: load score file ---
  score_map = load_score_file(score_file)
  if score_map is not None:
    print(f'[stage_c] loaded score file: {score_file}  ({len(score_map)} tokens, kappa={kappa})')
  else:
    print(f'[stage_c] no score file (baseline fixed-R, R(t)=R_base)')
  # --- end Stage C ---

  nusc = NuScenes(version=version, dataroot=data_root, verbose=True)

  results = {}

  total_time = 0.0
  total_frames = 0

  with open(detection_file) as f:
    data = json.load(f)
  assert 'results' in data, 'Error: No field `results` in result file. Please note that the result format changed.' \
    'See https://www.nuscenes.org/object-detection for more information.'

  all_results = EvalBoxes.deserialize(data['results'], DetectionBox)
  meta = data['meta']
  print('meta: ', meta)
  print("Loaded results from {}. Found detections for {} samples."
    .format(detection_file, len(all_results.sample_tokens)))

  processed_scene_tokens = set()
  for sample_token_idx in tqdm(range(len(all_results.sample_tokens))):
    sample_token = all_results.sample_tokens[sample_token_idx]
    scene_token = nusc.get('sample', sample_token)['scene_token']
    if scene_token in processed_scene_tokens:
      continue
    first_sample_token = nusc.get('scene', scene_token)['first_sample_token']
    current_sample_token = first_sample_token

    mot_trackers = {tracking_name: AB3DMOT(covariance_id, tracking_name=tracking_name, use_angular_velocity=use_angular_velocity, tracking_nuscenes=True) for tracking_name in NUSCENES_TRACKING_NAMES}

    while current_sample_token != '':
      results[current_sample_token] = []
      dets = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
      info = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
      for box in all_results.boxes[current_sample_token]:
        if box.detection_name not in NUSCENES_TRACKING_NAMES:
          continue
        q = Quaternion(box.rotation)
        angle = q.angle if q.axis[2] > 0 else -q.angle
        #[h, w, l, x, y, z, rot_y]
        detection = np.array([
          box.size[2], box.size[0], box.size[1],
          box.translation[0],  box.translation[1], box.translation[2],
          angle])
        information = np.array([box.detection_score])
        dets[box.detection_name].append(detection)
        info[box.detection_name].append(information)

      dets_all = {tracking_name: {'dets': np.array(dets[tracking_name]), 'info': np.array(info[tracking_name])}
        for tracking_name in NUSCENES_TRACKING_NAMES}

      # --- Stage C: look up S_t for this frame, apply to every active tracker ---
      if score_map is not None:
        current_s_t = score_map.get(current_sample_token, 1.0)
      else:
        current_s_t = 1.0  # baseline: R(t) = R_base
      for tracking_name in NUSCENES_TRACKING_NAMES:
        mot_trackers[tracking_name]._current_s_t = current_s_t
        mot_trackers[tracking_name]._kappa = kappa
      # --- end Stage C ---

      total_frames += 1
      start_time = time.time()
      for tracking_name in NUSCENES_TRACKING_NAMES:
        if dets_all[tracking_name]['dets'].shape[0] > 0:
          trackers = mot_trackers[tracking_name].update(dets_all[tracking_name], match_distance, match_threshold, match_algorithm, scene_token)
          # (N, 9)
          # (h, w, l, x, y, z, rot_y), tracking_id, tracking_score
          for i in range(trackers.shape[0]):
            sample_result = format_sample_result(current_sample_token, tracking_name, trackers[i])
            results[current_sample_token].append(sample_result)
      cycle_time = time.time() - start_time
      total_time += cycle_time

      # get next frame and continue the while loop
      current_sample_token = nusc.get('sample', current_sample_token)['next']

    # left while loop and mark this scene as processed
    processed_scene_tokens.add(scene_token)

  # finished tracking all scenes, write output data
  output_data = {'meta': meta, 'results': results}
  with open(output_path, 'w') as outfile:
    json.dump(output_data, outfile)

  print("Total Tracking took: %.3f for %d frames or %.1f FPS"%(total_time,total_frames,total_frames/total_time))
  print(f'[stage_c] output -> {output_path}')


def parse_args():
  p = argparse.ArgumentParser(description='Stage C tracker (vendored Chiu + R(t))')
  # original positional args (kept for compatibility with run.sh-style invocation)
  p.add_argument('data_split', type=str, help='val')
  p.add_argument('covariance_id', type=int, help='2 for nuscenes')
  p.add_argument('match_distance', type=str, help='m (mahalanobis) or iou')
  p.add_argument('match_threshold', type=float, help='11 for mahalanobis')
  p.add_argument('match_algorithm', type=str, help='greedy or h')
  p.add_argument('use_angular_velocity', type=str, help='true or false')
  p.add_argument('dataset', type=str, help='nuscenes')
  p.add_argument('save_root', type=str, help='output dir')
  # Stage C extensions
  p.add_argument('--detection-file', type=str, default=None,
                 help='path to results_nusc.json (BEVFusion detections)')
  p.add_argument('--data-root', type=str, default='/workspace/mmdetection3d/data/nuscenes',
                 help='nuScenes dataroot')
  p.add_argument('--version', type=str, default='v1.0-trainval')
  p.add_argument('--output', type=str, default=None,
                 help='output tracking JSON path')
  p.add_argument('--score-file', type=str, default=None,
                 help='path to scores_bev_real/<condition>_scores.npz, or None for baseline')
  p.add_argument('--kappa', type=float, default=3.0,
                 help='R(t) inflation coefficient (sweep {1,3,5})')
  return p.parse_args()


if __name__ == '__main__':
  args = parse_args()
  use_angular_velocity = args.use_angular_velocity == 'True' or args.use_angular_velocity == 'true'

  if args.dataset == 'kitti':
    print('track kitti not supported')
  elif args.dataset == 'nuscenes':
    print('track nuscenes')
    track_nuscenes(
      args.data_split, args.covariance_id, args.match_distance,
      args.match_threshold, args.match_algorithm, args.save_root,
      use_angular_velocity,
      detection_file=args.detection_file,
      data_root=args.data_root,
      version=args.version,
      output_path=args.output,
      score_file=args.score_file,
      kappa=args.kappa,
    )