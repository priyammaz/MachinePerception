import numpy as np
from matching_utils import iou

def data_association(dets, trks, threshold=0.2, algm='greedy'):
    """
    Q1. Assigns detections to tracked object

    dets:       a list of Box3D object
    trks:       a list of Box3D object
    threshold:  only mark a det-trk pair as a match if their iou distance is less than the threshold
    algm:       for extra credit, implement the hungarian algorithm as well

    Returns 3 lists:
        matches, kx2 np array of match indices, i.e. [[det_idx1, trk_idx1], [det_idx2, trk_idx2], ...]
        unmatched_dets, a 1d array of indices of unmatched detections
        unmatched_trks, a 1d array of indices of unmatched trackers
    """
    # Hint: you should use the provided iou(box_a, box_b) function to compute distance/cost between pairs of box3d objects
    # iou() is an implementation of a 3D box IoU

    matches = []
    unmatched_dets = []
    unmatched_trks = []

    if len(trks) == 0:
        unmatched_dets = np.array(list(range(len(dets))))
        return np.array(matches), np.array(unmatched_dets), np.array(unmatched_trks)
    else:
        dets_dict = {i: box for i, box in enumerate(dets)}
        trks_dict = {i: box for i, box in enumerate(trks)}

        while (len(dets_dict) != 0):
            dets_dict_copy = dets_dict.copy()
            trks_dict_copy = trks_dict.copy()
            for i, det in dets_dict_copy.items():
                best_iou = -np.inf
                best_j = 0
                for j, trk in trks_dict_copy.items():
                    iou_calc = iou(det, trk)
                    if iou_calc > best_iou:
                        best_iou = iou_calc
                        best_j = j
                if best_iou <= threshold:
                    unmatched_dets.append(i)
                    dets_dict.pop(i)
                    break
                else:
                    matches.append([i, best_j])
                    dets_dict.pop(i)
                    trks_dict.pop(best_j)
                    break

        unmatched_dets = np.array(unmatched_dets)
        unmatched_trks = np.array(list(trks_dict.keys()))
        matches = np.array(matches)

        return matches, unmatched_dets, unmatched_trks
# --------------------------- End your code here   ---------------------------------------------
