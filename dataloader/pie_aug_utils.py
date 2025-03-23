import numpy as np
import torch
import cv2

def proj_lidar2img(points, lidar2img, img_size=(1600, 900), min_dist=1.0, return_original=False):
    """Project lidar points to image plane.
    Args:
        points (numpy.array): Lidar points. (N, 3)
        lidar2img (numpy.array): Lidar to image matrix. (4, 4)
        img_size (tuple): Image size.
        min_dist (float): Minimum distance to the camera.
    """
    assert img_size[0] > img_size[1], 'img_size should be (W, H)'
    
    N = points.shape[0]
    if N == 0:
        return np.zeros((0, 2)), np.zeros(0, dtype=bool)

    points = np.concatenate([points, np.ones((N, 1))], axis=1)
    points_img = (lidar2img @ points.T)
    depths = points_img[2, :]
    points_img = points_img / points_img[2]
    img_W, img_H = img_size
    
    mask = np.ones(N, dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points_img[0, :] > 1)
    mask = np.logical_and(mask, points_img[0, :] < img_W - 1)
    mask = np.logical_and(mask, points_img[1, :] > 1)
    mask = np.logical_and(mask, points_img[1, :] < img_H - 1)

    if not return_original:
        points_img = points_img[:, mask]
    return points_img[:2, :].T, mask

def merge_images_pitch_torch(im1, im2, point_block):
    """merge two images with pitch angle according to point block"""
    assert point_block.shape[1] == 2, 'point block should be (n, 2)'
    if isinstance(im1, np.ndarray):
        im1 = torch.from_numpy(im1).to(point_block.device)
    if isinstance(im2, np.ndarray):
        im2 = torch.from_numpy(im2).to(point_block.device)        
    t, b = fit_to_box(point_block.cpu().numpy(), 'horizontal')
    # print(f'top: {t}, bottom: {b}')
    orig = im1.clone()
    new = im2.clone()
    # print(f'orig shape: {orig.shape}, new shape: {new.shape}')
    mask = torch.zeros_like(orig)
    orig[t:b,:,:] = new[t:b,:,:]
    mask[t:b,:,:] = 1
    # plt.imshow(orig[:,:,[2,1,0]])
    # plt.show()
    return orig, mask


def merge_images_pitch_np(im1, im2, point_block):
    """
    使用纯 numpy 操作，将 im2 在 [t:b, :, :] 范围内覆盖到 im1 上。
    覆盖范围由 point_block 的坐标在垂直方向(行)上的最小和最大值决定。
    
    参数:
        im1 (np.ndarray): 形状 (H, W, C) 的图像（如原始图像）
        im2 (np.ndarray): 形状 (H, W, C) 的图像（如混合图像）
        point_block (np.ndarray): 形状 (N, 2)，其中每行是 [x, y] 像素坐标
        
    返回:
        merged_im (np.ndarray): 合并后的图像
        mask (np.ndarray):      与 merged_im 形状相同的掩码，在 [t:b, :, :] 范围内为 1，其余为 0
    """
    assert isinstance(im1, np.ndarray), "im1 must be a numpy array"
    assert isinstance(im2, np.ndarray), "im2 must be a numpy array"
    assert point_block.shape[1] == 2, "point_block should be (N, 2)"
    
    # 获取覆盖区域的上下边界 (t: top, b: bottom)
    t, b = fit_to_box(point_block, 'horizontal')
    # print(f'top: {t}, bottom: {b}')
    # 为避免直接修改 im1，先复制一份
    merged_im = im1.copy()
    
    # 创建与图像同形状的掩码
    mask = np.zeros_like(merged_im, dtype=merged_im.dtype)
    
    # 将 im2 在 [t:b, :, :] 范围内覆盖到 merged_im
    # 请确保 t、b 不越界 (fit_to_box 通常会在内部进行 min/max 裁剪)
    merged_im[t:b, :, :] = im2[t:b, :, :]
    h, w = merged_im.shape[:2]
    # draw_dashed_line(merged_im, (0, t), (w, t), (1, 0, 0), 2)
    # draw_dashed_line(merged_im, (0, b), (w, b), (0, 1, 0), 2)
    # 对应区域的 mask 置为 1
    mask[t:b, :, :] = 1
    
    return merged_im, mask

def merge_images_yaw(im1, im2, l, r):
    """merge the mixed image to the right side of the original image"""
    assert im1.shape == im2.shape
    assert l <= r
    
    l, r = int(l), int(r)
    W = im1.shape[1]
    
    new = np.array(im2).copy()
    orig = np.array(im1).copy()
    
    new[:, l:r, :] = orig[:, l:r, :]
    return new

def fit_to_box(points, mode):
# def draw_box_boundary(points, mode):
    box = fit_box_cv(points)
    if   mode == 'vertical':   k = 0
    elif mode == 'horizontal': k = 1
    else: raise ValueError('mode should be vertical or horizontal')
    
    sorted_by_x = sorted(box, key=lambda p: p[k]) # if 'vertical': left&right sides, if 'horizontal': top&bottom sides
    side1 = sorted_by_x[:2]  
    side1_mid = np.mean(side1, axis=0)  

    side2 = sorted_by_x[-2:] 
    side2_mid = np.mean(side2, axis=0) 
    return side1_mid[k].astype(int), side2_mid[k].astype(int)

def fit_box_cv(points):
    # calculate the box
    rect = cv2.minAreaRect(points.astype(np.float32))  # Convert data type to CV_32F
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    return box

def draw_dashed_line(img, pt1, pt2, color, thickness=2, dash_length=10):
    """
    Draw a dashed line on an image.
    """
    if isinstance(pt1, list) or isinstance(pt1, tuple):
        pt1 = np.array(pt1)
    if isinstance(pt2, list) or isinstance(pt2, tuple):
        pt2 = np.array(pt2)
    if not isinstance(img, np.ndarray):
        img_np = img.cpu().numpy()
    else:
        img_np = img
        
    dist = np.linalg.norm(np.array(pt2) - np.array(pt1))
    dashes = np.linspace(0, dist, int(dist / dash_length))
    
    for i in range(0, len(dashes) - 1, 2):
        start = pt1 + (pt2 - pt1) * (dashes[i] / dist)
        end = pt1 + (pt2 - pt1) * (dashes[i + 1] / dist)
        start = tuple(start.astype(int))
        end = tuple(end.astype(int))
        img_np = cv2.line(img_np, start, end, color, thickness)
    
    if not isinstance(img, np.ndarray):
        img = torch.from_numpy(img_np)
    else:
        img = img_np
    return img

def draw_dashed_box(img, box, color=(0, 0, 0), thickness=2, dash_length=10):
    """
    绘制虚线框的方法
    :param img: 图像
    :param box: 矩形的四个顶点坐标
    :param color: 颜色
    :param thickness: 线条厚度
    :param dash_length: 每个虚线段的长度
    """
    # 绘制矩形的四条边
    for i in range(4):
        pt1 = box[i]
        pt2 = box[(i + 1) % 4]  # 每次连接当前点和下一个点，最后一条边连接到第一个点
        img = draw_dashed_line(img, pt1, pt2, color, thickness, dash_length)
    return img

def fit_to_box(points, mode):
# def draw_box_boundary(points, mode):
    box = fit_box_cv(points)
    if   mode == 'vertical':   k = 0
    elif mode == 'horizontal': k = 1
    else: raise ValueError('mode should be vertical or horizontal')
    
    sorted_by_x = sorted(box, key=lambda p: p[k]) # if 'vertical': left&right sides, if 'horizontal': top&bottom sides
    side1 = sorted_by_x[:2]  
    side1_mid = np.mean(side1, axis=0)  

    side2 = sorted_by_x[-2:] 
    side2_mid = np.mean(side2, axis=0) 
    return side1_mid[k].astype(int), side2_mid[k].astype(int)

def expand_box(box, x_expand, image_shape):
    # 假设 box 是由 cv2.boxPoints(rect) 生成的四个顶点
    # image_shape 是图像的形状 (height, width)

    # 获取图像的高度和宽度
    img_height, img_width = image_shape[:2]
    
    # 获取 box 的左、右、上、下边界的 x 和 y 坐标
    x_coords = box[:, 0]
    y_coords = box[:, 1]

    # 分别拓展四个边界
    min_x = max(np.min(x_coords) - x_expand, 0)  # 左边界拓展
    max_x = min(np.max(x_coords) + x_expand, img_width - 1)  # 右边界拓展
    min_y = max(np.min(y_coords) - x_expand, 0)  # 上边界拓展
    max_y = min(np.max(y_coords) + x_expand, img_height - 1)  # 下边界拓展

    # 生成拓展后的矩形四个顶点 (假设是轴对齐矩形)
    expanded_box = np.array([[min_x, min_y],
                             [max_x, min_y],
                             [max_x, max_y],
                             [min_x, max_y]], dtype=np.int32)
    
    return expanded_box

def crop_box_img(box, img):
    assert isinstance(img, np.ndarray)
    assert box.min() >= 0
    x, y, w, h = cv2.boundingRect(box)
    cropped_img = img[y:y+h, x:x+w]
    return cropped_img

def paste_box_img(box, source_img, target_img):
    assert isinstance(source_img, np.ndarray)
    assert isinstance(target_img, np.ndarray)
    assert box.min() >= 0
    x, y, w, h = cv2.boundingRect(box)
    if not source_img[y:y+h, x:x+w].shape == target_img.shape:
        # resize the target image
        target_img = cv2.resize(target_img, (w, h))
    source_img[y:y+h, x:x+w] = target_img
    return source_img


