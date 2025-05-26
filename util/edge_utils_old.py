import numpy as np
import cv2
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry import LineString
from shapely.ops import unary_union
def remove_rooms_with_iou(polygon_list):
    # Compute the IOU between each pair
    room_map_list = []
    for room_ind, poly in enumerate(polygon_list):
        room_map = np.zeros((256, 256))
        cv2.fillPoly(room_map, [np.array(poly.exterior.coords, dtype=np.int32)[:-1]], color=1.)
        room_map_list.append(room_map)

    access_mat = np.zeros((len(polygon_list), len(polygon_list)))
    remove_indices = []
    for idx0, polygon0 in enumerate(polygon_list):
        for idx1, polygon1 in enumerate(polygon_list):
            if idx0 == idx1 or access_mat[idx0][idx1] == 1 or access_mat[idx1][idx0] == 1:
                continue
            # compute the iou bewteen polygon0 and polygon1
            intersection = ((room_map_list[idx0] + room_map_list[idx1]) == 2)
            union = ((room_map_list[idx0] + room_map_list[idx1]) >= 1)
            iou = np.sum(intersection) / (np.sum(union) + 1)
            if iou > 0.4:
                print("移除iou")
                remove_indices.append([idx0, idx1])
            # print(f'idx_0: {idx0}, idx1: {idx1}, iou: {iou}')
            access_mat[idx0][idx1] = 1
            access_mat[idx1][idx0] = 1
    
    for remove_index in remove_indices:
        idx0, idx1 = remove_index[0], remove_index[1]
        polygon0, polygon1 = polygon_list[idx0], polygon_list[1]
        poly_area0, poly_area1 = polygon0.area, polygon1.area
        if poly_area0 > poly_area1:
            del polygon_list[idx1]
        else:
            del polygon_list[idx0]
    return polygon_list
def refine_rooms(polygon_list,overlap):
    access_mat = np.zeros((len(polygon_list), len(polygon_list)))
    for idx0, polygon0 in enumerate(polygon_list):
        for idx1, polygon1 in enumerate(polygon_list):
            if idx0 == idx1 or access_mat[idx0][idx1] == 1 or access_mat[idx1][idx0] == 1:
                continue
            if polygon0.intersects(polygon1):
                intersection = polygon0.intersection(polygon1)
                intersection_area = intersection.area
                if intersection_area >= 20:
                    overlap = True
                    # remove intersection from larger polygon
                    area0, area1 = polygon0.area, polygon1.area

                    if area0 > area1 and area1 > intersection_area:
                        polygon0 = polygon0.difference(intersection)
                        polygon1 = polygon1.union(intersection)
                    elif area0 < area1 and area0 > intersection_area:
                        polygon1 = polygon1.difference(intersection)
                        polygon0 = polygon0.union(intersection)
                    elif area0 > area1 and area1 == intersection_area:
                        polygon0 = polygon0.difference(intersection)
                        polygon1 = polygon1.union(intersection)

                        if polygon0.geom_type == 'MultiPolygon':
                            polygon_num = len(polygon0.geoms)
                            max_area = 0
                            smaller_polygon = None
                            for _ in range(polygon_num):
                                area = polygon0.geoms[_].area
                                if area > max_area:
                                    smaller_polygon = polygon0.geoms[_]
                                    max_area = area
                            polygon0 = smaller_polygon
                    elif area0 < area1 and area0 == intersection_area:
                        polygon1 = polygon1.difference(intersection)
                        polygon0 = polygon0.union(intersection)
                        if polygon1.geom_type == 'MultiPolygon':
                            polygon_num = len(polygon1.geoms)
                            max_area = 0
                            smaller_polygon = None
                            for _ in range(polygon_num):
                                area = polygon1.geoms[_].area
                                if area > max_area:
                                    smaller_polygon = polygon1.geoms[_]
                                    max_area = area
                            polygon1 = smaller_polygon
                    polygon_list[idx0] = polygon0
                    polygon_list[idx1] = polygon1
            access_mat[idx0][idx1] = 1
            access_mat[idx1][idx0] = 1
    return polygon_list,overlap
def merge_points(points, threshold=10):
    merged_points = points.copy()

    dp_points = []
    for i in range(merged_points.shape[0]):
        dp_points.append([merged_points[i]])
    dp_points = np.array(dp_points)
    try:
        dp_points = cv2.approxPolyDP(dp_points, epsilon=threshold, closed=True)
        merged_points = []
        for i in range(dp_points.shape[0]):
            merged_points.append(dp_points[i][0])
        merged_points = np.array(merged_points)
        return merged_points
    except:
        return points


    
def line_distance(line1: np.ndarray, line2: np.ndarray) -> np.ndarray:
    """精确计算平行线段间最小距离（修复广播问题）"""
    # 提取线段端点
    p1, p2 = line1[:, :2], line1[:, 2:]
    q1, q2 = line2[:, :2], line2[:, 2:]

    # 计算当前线段方向向量
    dir_vec = p2 - p1
    norm_dir = np.linalg.norm(dir_vec, axis=1, keepdims=True) + 1e-8
    unit_dir = dir_vec / norm_dir #第一条线段的单位方向向量

    # 计算q端点的投影参数
    vec_pq1 = q1 - p1
    vec_pq2 = q2 - p1
    proj1 = np.sum(vec_pq1 * unit_dir, axis=1)  # shape (n,)
    proj2 = np.sum(vec_pq2 * unit_dir, axis=1)  # shape (n,)

    # 确定投影区间边界
    proj_min = np.minimum(proj1, proj2)
    proj_max = np.maximum(proj1, proj2)

    # 截断到有效区间[0, norm_dir]
    clamped_min = np.maximum(0, proj_min)
    clamped_max = np.minimum(norm_dir.squeeze(), proj_max)

    # 计算最近点（维度修正关键点）
    closest_p = p1 + unit_dir * clamped_min[:, None]  # 添加维度适配广播

    # 修复维度问题：使用布尔索引选择q端点
    mask = (proj_min == proj1)  # True表示q1是更近端点
    closest_q = np.where(mask[:, None], q1, q2)  # 添加维度适配广播

    # 计算最小距离
    delta = closest_p - closest_q
    return np.sqrt(np.sum(delta ** 2, axis=1))
def detect_duplicate_edges(edges: np.ndarray,
                           angle_threshold: float = 1.0,
                           distance_threshold: float = 5.0) -> np.ndarray:
    """检测并标记重复边（矩阵运算优化版）"""
    n = len(edges)
    if n < 2:
        return np.zeros(n, dtype=bool)

    # 生成环形边对
    next_edges = np.roll(edges, -1, axis=0)

    # 计算边向量
    current_vec = edges[:, 2:] - edges[:, :2]
    next_vec = next_edges[:, 2:] - next_edges[:, :2]

    # 平行判断
    parallel_mask = is_parallel(current_vec, next_vec, angle_threshold)

    # 计算相邻边间距
    dist_mask = line_distance(edges, next_edges) <= distance_threshold

    # 综合判断条件
    duplicate_mask = parallel_mask & dist_mask

    # 生成保留标记（每组重复边保留第一条）
    keep_mask = np.ones(n, dtype=bool)
    for i in range(n):
        if duplicate_mask[i]:
            # 比较边长度，保留较长者
            len_current = np.linalg.norm(current_vec[i])
            len_next = np.linalg.norm(next_vec[i])
            if len_next > len_current:
                keep_mask[i] = False
            else:
                keep_mask[(i + 1) % n] = False
    return keep_mask
def remove_short_edges(edges, threshold=5):
    
    dx = edges[:, 2] - edges[:, 0]  # x2 - x1
    dy = edges[:, 3] - edges[:, 1]  # y2 - y1
    
    dist_sq = dx ** 2 + dy ** 2
    threshold_sq = threshold ** 2
    mask = dist_sq > threshold_sq
    
    filtered_edges = edges[mask]
    return filtered_edges
def is_parallel(vec1: np.ndarray, vec2: np.ndarray, angle_threshold: float = 5.0) -> np.ndarray:
    """改进的平行判断：结合方向向量夹角和距离阈值"""
    # 计算单位向量
    norm1 = np.linalg.norm(vec1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(vec2, axis=1, keepdims=True)
    unit_vec1 = vec1 / (norm1 + 1e-8)
    unit_vec2 = vec2 / (norm2 + 1e-8)

    # 计算夹角余弦值
    cos_theta = np.sum(unit_vec1 * unit_vec2, axis=1)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi

    # 综合判断条件
    return (theta < angle_threshold) | (theta > (180 - angle_threshold))
def compute_intersections_matrix(edges, threshold=10):
    """ 批量计算相邻边交点 (矩阵运算版本) """
    starts = edges[:,:2]
    ends = edges[:,2:]
    n = len(edges)

    # 计算相邻边间距
    next_starts = np.roll(starts, -1, axis=0)
    next_ends = np.roll(ends, -1, axis=0)
    dist_sq = np.sum((ends - next_starts) ** 2, axis=1)
    mask = dist_sq <= threshold ** 2
    if ~mask[-1]:
        mask[-1] = True    


    # 提取需要计算的边对
    A = starts[mask]  # 前一条边的起点
    B = ends[mask]  # 前一条边的终点
    C = next_starts[mask]  # 后一条边的起点
    D = next_ends[mask]  # 后一条边的终点

    # 向量计算
    AB = B - A
    CD = D - C
    AC = C-A
    parallel_mask = is_parallel(AB, CD)

    # 矩阵行列式计算
    det = AB[:, 0] * CD[:, 1] - AB[:, 1] * CD[:, 0]
    abs = np.abs(det)
    valid = np.abs(det) > 1e-6

    # 参数计算
    t = (CD[:, 1] * AC[:, 0] - CD[:, 0] * AC[:, 1]) / (det + 1e-6)
    s = (AB[:, 1] * AC[:, 0] - AB[:, 0] * AC[:, 1]) / (det + 1e-6)
    # 计算交点坐标
    intersections = A + t[:, None] * AB
    valid = valid  & (t >= 0) & (t <= 1.5) & (s >= -1e6) & (s <= 1e6) &(~parallel_mask)  # 限制合理范围
    return mask, intersections, valid

def remove_multi_polygon(polygon_lst):
    for poly_idx, polygon in enumerate(polygon_lst):
        connect_edges = []
        if isinstance(polygon, MultiPolygon):
            poly_eqs = []
            poly_pts = []
            poly_v = []
            print("remove multi polygon",len(polygon.geoms),len(polygon_lst))
            #遍历multipolygon的每一个多边形
            for sub_polygon in polygon.geoms:
                print("remove multi polygon")
                # There may exists some exterior corners in the vertices, remove them
                poly_np = np.array(sub_polygon.exterior.coords, dtype=np.uint8)[:-1]
                simplified_poly_np = simplify_polygon(poly_np)
                poly_np = simplified_poly_np
                poly_eq = []
                poly_pt = []
                for i in range(len(poly_np)-1):
                    print(i)
                    start_point = poly_np[i]
                    end_point = poly_np[(i + 1) % len(poly_np)]
                    if start_point[0] == end_point[0]:  # Vertical line
                        line_eq = [float('inf'), start_point[0]]
                    elif start_point[1] == end_point[1]:  # Horizontal line
                        line_eq = [0, start_point[1]]
                    else:
                        #拟合出两个点之间的线段
                        line_eq = np.polyfit([start_point[0], end_point[0]], [start_point[1], end_point[1]], 1)
                    poly_eq.append(line_eq)
                    poly_pt.append([start_point, end_point])
                    
                poly_v.append(poly_np)
                poly_eqs.append(poly_eq)
                poly_pts.append(poly_pt)
            
            assert len(poly_v) == len(poly_eqs) == 2
            

            for i, poly_eq in enumerate(poly_eqs):
                src_polygon = poly_v[i]
                tgt_polygon = poly_v[(i+1)%len(poly_v)]
                poly_pt = poly_pts[i]
                for eq_i, eq in enumerate(poly_eq):
                    # Create a line based on the equation
                    pt = poly_pt[eq_i]

                    if eq[0] == float('inf'):  # Vertical line
                        line = LineString([(eq[1], 0), (eq[1], 255)])
                    elif eq[0] == 0:  # Horizontal line
                        line = LineString([(0, eq[1]), (255, eq[1])])
                    else:  # Diagonal line
                        line = LineString([(0, eq[1]), (255, eq[0] * 255 + eq[1])])
                    
                    intersection = line.intersection(Polygon(tgt_polygon))

                    # If there is an intersection, return the first intersection point
                    if not intersection.is_empty:
                        if intersection.geom_type == 'Point':
                            intersection_point = (intersection.x, intersection.y)
                        elif intersection.geom_type == 'MultiPoint':
                            intersection_point = (intersection[0].x, intersection[0].y)
                        elif intersection.geom_type == 'LineString':
                            intersection_point = [(intersection.coords[0][0], intersection.coords[0][1]), 
                                                  (intersection.coords[1][0], intersection.coords[1][1])]
                        else:
                            continue

                        # Calculate the distance between points in pt and intersection_point
                        distances = []
                        for p in pt:
                            for ip in intersection_point:
                                distance = np.sqrt((p[0] - ip[0])**2 + (p[1] - ip[1])**2)
                                distances.append((distance, p, ip))

                        # Find the pair with the minimum distance
                        min_distance, min_pt, min_ip = min(distances, key=lambda x: x[0])

                        # Check if the line segment intersects with src_polygon
                        line_segment = LineString([min_pt, min_ip])
                        if line_segment.intersection(Polygon(src_polygon)).geom_type == 'Point':
                            connect_edges.append([min_pt, min_ip])
            
            if len(connect_edges) == 1:
                merge_polygon = polygon[0] if polygon[0].area > polygon[1].area else polygon[1]
            else:
                edge_points = np.array([point for edge in connect_edges for point in edge], dtype=np.uint8)
                new_edge_points = []
                new_edge_points.append(edge_points[0])
                edge_points = np.delete(edge_points, 0, axis=0)
                while edge_points.shape[0] > 0:
                    for idx, point in enumerate(edge_points):
                        if point[0] == new_edge_points[-1][0] or point[1] == new_edge_points[-1][1]:
                            new_edge_points.append(point)
                            edge_points = np.delete(edge_points, idx, axis=0)
                            break
                edge_polygon = Polygon(new_edge_points)                
                merge_polygon = unary_union([Polygon(src_polygon), Polygon(tgt_polygon), edge_polygon])

            poly_np = np.array(merge_polygon.exterior.coords, dtype=np.uint8)[:-1]
            simplified_poly_np = simplify_polygon(poly_np)
            # poly_np = simplified_poly_np
            poly_np = np.concatenate([simplified_poly_np, simplified_poly_np[None, 0]])
            update_polygon = Polygon(poly_np)
            polygon_lst[poly_idx] = update_polygon

    for polygon in polygon_lst:
        assert polygon.geom_type == 'Polygon'
    return polygon_lst

def get_corners_from_edges(edges, threshold=10):
    """ 多边形边优化主函数 """
    if len(edges) < 3:
        return edges
    
    keep_mask = detect_duplicate_edges(edges, 5, 5)
    filtered_edges = edges#[keep_mask]
    # filtered_edges = edges
    # # 第二步：重新闭合多边形
    # if len(filtered_edges) >= 3:
    #     last_point = filtered_edges[-1, 2:]
    #     first_point = filtered_edges[0, :2]
    #     if np.linalg.norm(last_point - first_point) > 1e-6:
    #         filtered_edges[-1, 2:] = first_point
    # 第一步：计算所有需要合并的边对
    mask, intersections, valid = compute_intersections_matrix(filtered_edges, threshold)
    valid_indices = np.where(mask)[0]
    index_map = {orig_idx: arr_idx for arr_idx, orig_idx in enumerate(valid_indices)}

    # 第二步：构建新顶点序列
    corners = []
    n=len(filtered_edges)
    for i in range(len(filtered_edges)):
        current_end = filtered_edges[i, 2:]
        next_start = filtered_edges[(i + 1) % n, :2]

        # 当满足合并条件时添加交点
        if mask[i]:
            # 通过映射表找到valid中的位置
            arr_idx = index_map.get(i, -1)
            if arr_idx != -1 and valid[arr_idx]:
                corners.append(intersections[arr_idx])
            else:
                # 否则添加当前终点和下个起点
                if not corners or tuple(corners[-1]) != tuple(current_end):
                    corners.append(current_end)
                if tuple(next_start) != tuple(current_end):
                    corners.append(next_start)
        else:
            # 否则添加当前终点和下个起点
            if not corners or tuple(corners[-1]) != tuple(current_end) :
                corners.append(current_end)
            if tuple(next_start) != tuple(current_end):
                corners.append(next_start)



    corners = np.array(corners)

    return corners
def simplify_polygon(input_poly):
    def is_angle_change(p1, p2, p3):
        v1 = p2 - p1
        v2 = p3 - p2
        angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
        return np.abs(angle) > 1e-2  # Adjust the threshold as needed

    simplified_poly_np = []
    for i in range(len(input_poly)):
        if is_angle_change(input_poly[(i - 1)%len(input_poly)], input_poly[i], input_poly[(i + 1)%len(input_poly)]):
            simplified_poly_np.append(input_poly[i])
    # simplified_poly_np.append(poly_np[-1])
    simplified_poly_np = np.array(simplified_poly_np, dtype=np.uint8)
    
    return simplified_poly_np

def remove_duplicate_corners(polygon):
    
    simplified_poly_np = simplify_polygon(polygon)
   

    return simplified_poly_np