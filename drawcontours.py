import cv2
import numpy as np
#测试merge效果
def mergeClusters(frame, threshold1, threshold2):
    height, width, channel = frame.img.shape
    label = frame.labels
    img_hsi = frame.img_hsi
    img_hsi = np.array(img_hsi,dtype=float)
    img_countours = frame.img.copy()
    dx = [-1, -1, 0, 1, 1, 1, 0, -1]
    dy = [0, -1, -1, -1, 0, 1, 1, 1]
    for j in range(height):
        for k in range(width):
            nop = 0
            for i in range(8):
                x = k + dx[i]
                y = j + dy[i]
                if x > 0 and x < width and y > 0 and y < height:
                    if label[j][k] != label[y][x]:
                        nop = nop + 1
                        dist = np.linalg.norm(img_hsi[j][k] - img_hsi[y][x])
                        if dist < threshold1:
                            cent1 = frame.sp_position[label[j][k]]
                            cent2 = frame.sp_position[label[y][x]]
                            centerdist = np.linalg.norm(img_hsi[cent1[0]][cent1[1]] - img_hsi[cent2[0]][cent2[1]])
                            if centerdist < threshold2:
                                img_countours[j][k] = [0, 255, 0]
                                nop = 0
                                continue
            if nop > 1:
                img_countours[j][k] = [0, 0, 255]
    return img_countours

# def dealwithedge()


# def DrawContoursAroundSegments(img,label):
#     dx = [-1, -1, 0, 1, 1, 1, 0, -1]
#     dy = [0, -1, -1, -1, 0, 1, 1, 1]
#     height, width, channel = img.shape
#     img_countours = img.copy()
#     for j in range(height):
#         for k in range(width):
#             np = 0
#             for i in range(8):
#                 x = k + dx[i]
#                 y = j + dy[i]
#                 if x > 0 and x < width and y > 0 and y < height:
#                     if label[j][k] != label[y][x]:
#                         np = np +1
#             if np > 1:
#                 img_countours[j, k] = [0, 0, 0]
#     cv2.imshow("draw contour", img_countours)
