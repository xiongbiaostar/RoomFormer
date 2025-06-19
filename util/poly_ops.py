"""
Utilities for polygon manipulation.
"""
import torch
import numpy as np

def get_polygon_vertices_matrix(edges):
    """ 用矩阵运算计算按顺序给出的边的角点 """
    #edges = np.array(edges).reshape(-1, 4)  # 确保是 (N, 4)
    N = len(edges)

    # 获取所有边的起点和终点
    x1, y1, x2, y2 = edges[:, 0], edges[:, 1], edges[:, 2], edges[:, 3]

    # 计算方向向量
    dx1, dy1 = x2 - x1, y2 - y1  # 当前边方向
    dx2, dy2 = np.roll(dx1, -1), np.roll(dy1, -1)  # 下一个边的方向
    x3, y3 = np.roll(x1, -1), np.roll(y1, -1)  # 下一个边的起点

    # 构造矩阵方程 Ax = B
    A = np.stack([-dx1, dx2, -dy1, dy2], axis=-1).reshape(N, 2, 2)
    B = np.stack([x1 - x3, y1 - y3], axis=-1).reshape(N, 2, 1)

    # 解线性方程组
    try:
        T = np.linalg.solve(A, B)  # 计算 t1, t2
        t1 = T[:, 0, 0]

        # 计算交点 (x, y) = (x1 + t1 * dx1, y1 + t1 * dy1)
        intersections = np.stack([x1 + t1 * dx1, y1 + t1 * dy1], axis=-1)

        return intersections
    except np.linalg.LinAlgError:
        return np.array([])  # 处理奇异矩阵情况（平行边）
def is_clockwise(points):
    """Check whether a sequence of points is clockwise ordered
    """
    # points is a list of 2d points.
    assert len(points) > 0
    s = 0.0
    for p1, p2 in zip(points, points[1:] + [points[0]]):
        s += (p2[0] - p1[0]) * (p2[1] + p1[1])
    return s > 0.0

def resort_corners(corners):
    """Resort a sequence of corners so that the first corner starts
       from upper-left and counterclockwise ordered in image
    """
    corners = corners.reshape(-1, 2)
    x_y_square_sum = corners[:,0]**2 + corners[:,1]**2 
    start_corner_idx = np.argmin(x_y_square_sum)

    corners_sorted = np.concatenate([corners[start_corner_idx:], corners[:start_corner_idx]])

    ## sort points clockwise (counterclockwise in image)
    if not is_clockwise(corners_sorted[:,:2].tolist()):
        corners_sorted[1:] = np.flip(corners_sorted[1:], 0)

    return corners_sorted.reshape(-1)


def get_all_order_corners(corners):
    """Get all possible permutation of a polygon
    """
    length = int(len(corners) / 4)
    all_corners = torch.stack([corners.roll(i*4) for i in range(length)])
    return all_corners


def pad_gt_polys(gt_instances, num_queries_per_poly, device):
    """Pad the ground truth polygons so that they have a uniform length
    """

    room_targets = []
    # padding ground truth on-fly
    for gt_inst in gt_instances:
        room_dict = {}
        room_corners = []
        corner_labels = []
        corner_lengths = []

        for i, poly in enumerate(gt_inst.gt_masks.polygons):
            corners = torch.from_numpy(poly[0]).to(device)
            corners = torch.clip(corners, 0, 255) / 255
            corner_lengths.append(len(corners))

            corners_pad = torch.zeros(num_queries_per_poly*2, device=device)
            corners_pad[:len(corners)] = corners

            labels = torch.ones(int(len(corners)/2), dtype=torch.int64).to(device)
            labels_pad = torch.zeros(num_queries_per_poly, device=device)
            labels_pad[:len(labels)] = labels
            room_corners.append(corners_pad)
            corner_labels.append(labels_pad)

        room_dict = {
            'coords': torch.stack(room_corners), #[num_polys,80]
            'labels': torch.stack(corner_labels),
            'lengths': torch.tensor(corner_lengths, device=device),
            'room_labels': gt_inst.gt_classes
        }
        room_targets.append(room_dict)


    return room_targets


def pad_gt_polys_to_edges(gt_instances, num_queries_per_poly, device):
    """Pad the ground truth polygons so that they have a uniform length
    """

    room_targets = []
    # padding ground truth on-fly
    for gt_inst in gt_instances:
        room_dict = {}
        room_corners = []
        corner_labels = []
        corner_lengths = []

        for i, poly in enumerate(gt_inst.gt_masks.polygons):
            corners = torch.from_numpy(poly[0])
            corners = torch.clip(corners, 0, 255) / 255
            num_corners = len(corners) // 2
            corners = corners.view(num_corners, 2)
            edges = torch.zeros((num_corners, 2, 2))
            if num_corners>2:
                
                for i in range(num_corners):
                    next_index = (i + 1) % num_corners
                    edge_start = corners[i]
                    edge_end = corners[next_index]
                    edges[i] = torch.stack([edge_start, edge_end])
                
                edges = edges.view(-1).to(device)#[num_points]
            else:
                edges = corners.view(-1).to(device)
            corner_lengths.append(len(edges))
            

            edges_pad = torch.zeros(num_queries_per_poly*4, device=device)
            edges_pad[:len(edges)] = edges #[160]160=40*4

            labels = torch.ones(int(len(edges)/4), dtype=torch.int64).to(device) 
            labels_pad = torch.zeros(num_queries_per_poly, device=device) #[80]
            labels_pad[:len(labels)] = labels #[40]
            room_corners.append(edges_pad)
            corner_labels.append(labels_pad)
        room_labels = gt_inst.gt_classes.clone()
        room_labels[room_labels <= 15] = 0
        room_labels[room_labels == 16] = 1
        room_labels[room_labels == 17] = 2
        room_dict = {
            'coords': torch.stack(room_corners), #[num_polys,160]
            'labels': torch.stack(corner_labels),#[num_polys,40]
            'lengths': torch.tensor(corner_lengths, device=device),#[num_polys]
            'room_labels': room_labels
        }
        room_targets.append(room_dict)


    return room_targets



def get_gt_polys(gt_instances, num_queries_per_poly, device):
    room_targets = []
    # padding ground truth on-fly
    for gt_inst in gt_instances:
        room_dict = {}
        room_corners = []
        corner_labels = []
        corner_lengths = []

        for i, poly in enumerate(gt_inst.gt_masks.polygons):
            corners = torch.from_numpy(poly[0])
            corners = torch.clip(corners, 0, 255) / 255
            num_corners = len(corners) // 2
            corners = corners.view(num_corners, 2)
            edges = torch.zeros((num_corners, 2, 2))
            if num_corners>2: 
                for i in range(num_corners):
                    next_index = (i + 1) % num_corners
                    edge_start = corners[i]
                    edge_end = corners[next_index]
                    edges[i] = torch.stack([edge_start, edge_end])
                
                edges = edges.view(-1).to(device)
                
                
            else:
                edges = corners.view(-1).to(device)
            corner_lengths.append(len(edges))
            
            # corners_pad = torch.zeros(num_queries_per_poly*4, device=device)
            # corners_pad[:len(edges)] = edges

            labels = torch.ones(int(len(edges)/4), dtype=torch.int64).to(device) 
            # labels_pad = torch.zeros(num_queries_per_poly, device=device) #[80]
            # labels_pad[:len(labels)] = labels
            room_corners.append(edges)
            corner_labels.append(labels)
        room_labels = gt_inst.gt_classes.clone()
        room_labels[room_labels <= 15] = 0
        room_labels[room_labels == 16] = 1
        room_labels[room_labels == 17] = 2
        room_dict = {
            'coords': torch.cat(room_corners), #[num_edges_of_a_batch*4]
            'labels': torch.cat(corner_labels), #[num_edges_of_a_batch]
            'lengths': torch.tensor(corner_lengths, device=device),
            'room_labels': room_labels
        }
        room_targets.append(room_dict)


    return room_targets



