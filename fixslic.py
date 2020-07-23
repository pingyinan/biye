from basic_elements import Frame
from basic_elements import Cluster
import matplotlib.pyplot as plt
import numpy as np
import cv2
class fixslic_process:
    def __init__(self, frame):
        self.frame = frame

#按单个像素进行移动
    def startfixprocess(self, iter):
        self.updatelabels_pixels(iter)


    def updatelabels_pixels(self, iter):    #按像素
        for i in range(iter):
            for edge, points in self.frame.edges.items():
                c1, c2 = edge
                for point in points:
                    h, w = point
                    p_bin = self.frame.f_bin[h, w]
                    if self.frame.sp_hist[c1, p_bin] > self.frame.sp_hist[c2, p_bin]:  #需要更新cluster
                        self.frame.labels[h][w] = c1
                    else:
                        self.frame.labels[h][w] = c2
            self.frame.updatesp()   #更新sp_area、sp_position、sp_hist
            self.updateEdges()
            img_contour = self.frame.DrawContoursAroundSegments()
            cv2.imshow("iter{}".format(i),img_contour)
            cv2.waitKey(500)

    def updateEdges(self):
        frame = self.frame
        frame.edges = {}
        height, width, channel = frame.img.shape
        dx = [-1, -1, 0, 1, 1, 1, 0, -1]
        dy = [0, -1, -1, -1, 0, 1, 1, 1]
        for j in range(height):
            for k in range(width):
                for i in range(8):
                    x = k + dx[i]
                    y = j + dy[i]
                    if x > 0 and x < width and y > 0 and y < height:
                        if frame.labels[j][k] != frame.labels[y][x]:
                            c1 = min(frame.labels[j][k], frame.labels[y][x])
                            c2 = max(frame.labels[j][k], frame.labels[y][x])
                            frame.edges.setdefault((c1, c2), []).append((j, k))
        for edge in frame.edges:
            frame.edges[edge] = list(set(frame.edges[edge]))
        self.frame = frame

#
    def updatelabels_process(self, iter, scale):
        flag = True
        for i in range(iter):
            if flag:
                self.updatelabels_blocks(0, scale)
                flag = False
            else:
                self.updatelabels_blocks(scale/2, scale)
                flag = True
            img_contour = self.frame.DrawContoursAroundSegments()
            self.choosen = []
            def Mousecallback(event, x, y, flags, param):
                color = ['red', 'gold', 'darkgreen', 'blue', 'gray', 'darksalmon', 'olivedrab',
                         'lightseagreen', 'darkorchid', 'navy', 'm', 'rosybrown', 'firebrick',
                         'chartreuse', 'royalblue', 'plum', 'silver']
                scale = np.arange(513)
                flag = 0
                if event == cv2.EVENT_FLAG_LBUTTON:
                    flag = 1
                    self.startpoint = (x, y)
                    label = self.frame.labels[y][x]
                    self.choosen.append(label)
                    plt.figure()
                    plt.title("label:{}".format(label))
                    for i in range(len(self.choosen)):
                        hsi_hist = self.frame.sp_hist[self.choosen[i]]
                        plt.plot(scale, hsi_hist, color=color[i])
                    plt.show()
                if flag == 1 and event == cv2.EVENT_MOUSEMOVE:
                    self.currentpoint = (x, y)
                    imgshow = img_contour.copy()
                    cv2.rectangle(imgshow, self.startpoint, self.currentpoint, 'r')
                    cv2.imshow(winname, imgshow)
                if event == cv2.EVENT_LBUTTONUP:
                    flag = 0
                    self.endpoint = (x, y)
                    blockhist = np.zeros(513, dtype=np.int)
                    for h in range(self.startpoint[1], self.endpoint[1]):
                        for w in range(self.startpoint[0], self.endpoint[0]):
                            blockhist[self.frame.f_bin[h, w]] += 1
                    blockhist = blockhist / (
                                (self.startpoint[1] - self.endpoint[1]) * (self.startpoint[0] - self.endpoint[0]))
                    plt.figure()
                    plt.plot(scale, blockhist, 'r')
                    plt.title("choosen area")
                    plt.show()
                if event == cv2.EVENT_FLAG_RBUTTON:
                    self.choosen = []
                    cv2.imshow(winname, img_contour)

            winname = "iter{}".format(i)
            cv2.namedWindow(winname)
            cv2.setMouseCallback(winname, Mousecallback)
            cv2.imshow(winname, img_contour)
            cv2.waitKey(50)
            # if cv2.waitKey(0) == 'q':
            #     pass
        return self.frame


    def updatelabels_blocks(self, offset, scale):
        checkedBlocks = []
        # edges = [(358, 362), (245, 256), (362, 379)]
        height, width, channel = self.frame.img.shape
        for edge, edge_points in self.frame.edges.items():
        # for edge in edges:
            c1, c2 = edge
            edge_points = self.frame.edges[edge]
            for point in edge_points:
                h ,w = point
                h_index = int((h - offset)/scale)
                w_index = int((w - offset)/scale)
                if h - offset < 0:
                    h_index = -1
                if w - offset < 0:
                    w_index = -1
                if (h_index, w_index) in checkedBlocks:
                    continue
                else:
                    checkedBlocks.append((h_index, w_index))
                    if h_index == -1:
                        h_start = 0
                    else:
                        h_start = offset + scale * h_index
                    if w_index == -1:
                        w_start = 0
                    else:
                        w_start = offset + scale * w_index
                    h_end = h_start + scale
                    w_end = w_start + scale
                    if h_end > height:
                        h_end = height
                    if w_end > width:
                        w_end = width
                    c1_hist, c1_exr_hist, c1_ori_hist, c1_pixels, c2_hist, c2_exr_hist, c2_ori_hist, c2_pixels = self.calculate_hist(
                        c1, c2, h_start, w_start, h_end, w_end)
                    if self.Int(c1_hist, c1_exr_hist, c2_ori_hist):  # support move block
                        for h, w in c1_pixels:
                            self.frame.labels[h][w] = c2
                    elif self.Int(c2_hist, c2_exr_hist, c1_ori_hist):
                        for h, w in c2_pixels:
                            self.frame.labels[h][w] = c1
            checkedBlocks = []
        self.frame.updatesp()
        self.updateEdges()

    #仅测试245cluster的modify效果
    def updatelabels_245(self, iter):
        edges = [(358, 362), (245, 256),(362, 379)]
        for i in range(iter):
            for edge in edges:
                c1, c2 = edge
                edge_points = self.frame.edges[edge]
                for point in edge_points:
                    self.move_block(c1, c2, point, 8)
            self.frame.updatesp()   #更新sp_area、sp_position、sp_hist
            self.updateEdges()
            img_contour = self.frame.DrawContoursAroundSegments()
            self.choosen = []
            def Mousecallback(event, x, y, flags, param):
                color = ['red', 'gold', 'darkgreen', 'blue', 'gray', 'darksalmon', 'olivedrab',
                         'lightseagreen', 'darkorchid', 'navy', 'm', 'rosybrown', 'firebrick',
                         'chartreuse', 'royalblue', 'plum', 'silver']
                scale = np.arange(513)
                flag = 0
                if event == cv2.EVENT_FLAG_LBUTTON:
                    flag = 1
                    self.startpoint = (x, y)
                    label = self.frame.labels[y][x]
                    self.choosen.append(label)
                    plt.figure()
                    plt.title("label:{}".format(label))
                    for i in range(len(self.choosen)):
                        hsi_hist = self.frame.sp_hist[self.choosen[i]]
                        plt.plot(scale, hsi_hist, color=color[i])
                    plt.show()
                    # avr_hist = np.zeros(513, dtype=np.float)
                    # plt.figure()
                    # plt.subplot(1,2,1)
                    # for i in range(len(self.choosen)):
                    #     hsi_hist = self.choosen[i].hsi_hist
                    #     plt.plot(scale, hsi_hist, color=color[i])
                    #     avr_hist += np.array(hsi_hist)
                    # #plt.title("label:{}".format(cluster.index))
                    # plt.subplot(1,2,2)
                    # avr_hist = avr_hist/len(self.choosen)
                    # plt.plot(scale, avr_hist, 'r')
                    # plt.show()
                if flag == 1 and event == cv2.EVENT_MOUSEMOVE:
                    self.currentpoint = (x,y)
                    imgshow = img_contour.copy()
                    cv2.rectangle(imgshow, self.startpoint, self.currentpoint, 'r')
                    cv2.imshow(winname, imgshow)
                if event == cv2.EVENT_LBUTTONUP:
                    flag = 0
                    self.endpoint = (x, y)
                    blockhist = np.zeros(513, dtype=np.int)
                    for h in range(self.startpoint[1], self.endpoint[1]):
                        for w in range(self.startpoint[0], self.endpoint[0]):
                            blockhist[self.frame.f_bin[h, w]] += 1
                    blockhist = blockhist/((self.startpoint[1] - self.endpoint[1])* (self.startpoint[0] - self.endpoint[0]))
                    plt.figure()
                    plt.plot(scale, blockhist, 'r')
                    plt.title("choosen area")
                    plt.show()
                if event == cv2.EVENT_FLAG_RBUTTON:
                    self.choosen = []
                    cv2.imshow(winname, img_contour)
            winname = "iter{}".format(i)
            cv2.namedWindow(winname)
            cv2.setMouseCallback(winname, Mousecallback)
            cv2.imshow(winname,img_contour)
            if cv2.waitKey(0) == 'q':
                pass


    def move_block(self, c1, c2, mid, s):
        upleft_h = int(mid[0] - s / 2)
        upleft_w = int(mid[1] - s / 2)
        downright_h = upleft_h + s
        downright_w = upleft_w + s
        height, width, channel = self.frame.img.shape
        if upleft_w < 0:
            upleft_w = 0
        if upleft_h < 0:
            upleft_h = 0
        if downright_h > height - 1:
            downright_h = height - 1
        if downright_w > width - 1:
            downright_w = width - 1
        c1_hist, c1_exr_hist, c1_ori_hist, c1_pixels, c2_hist, c2_exr_hist, c2_ori_hist, c2_pixels = self.calculate_hist(c1, c2, upleft_h, upleft_w, downright_h, downright_w)
        if self.Int(c1_hist, c1_exr_hist, c2_ori_hist):  # support move block
            for h, w in c1_pixels:
                self.frame.labels[h][w] = c2
        elif self.Int(c2_hist, c2_exr_hist, c1_ori_hist):
            for h, w in c2_pixels:
                self.frame.labels[h][w] = c1

    def calculate_hist(self, c1, c2, upleft_h, upleft_w, downright_h, downright_w):
        c1_hist = np.zeros(513, dtype=np.float)
        c1_pixels = []
        c2_hist = np.zeros(513, dtype=np.float)
        c2_pixels = []

        for h in range(int(upleft_h), int(downright_h)):
            for w in range(int(upleft_w), int(downright_w)):
                if self.frame.labels[h][w] == c1:
                    c1_pixels.append((h, w))
                    c1_hist[self.frame.f_bin[h, w]] += 1
                if self.frame.labels[h][w] == c2:
                    c2_pixels.append((h, w))
                    c2_hist[self.frame.f_bin[h, w]] += 1
        c1_ori_hist = self.frame.sp_hist[c1]
        c1_ori_area = self.frame.sp_area[0, c1]
        c1_exr_hist = (c1_ori_hist * c1_ori_area - c1_hist) / (c1_ori_area - len(c1_pixels))
        c1_hist = c1_hist / len(c1_pixels)

        c2_ori_hist = self.frame.sp_hist[c2]
        c2_ori_area = self.frame.sp_area[0, c2]
        c2_exr_hist = (c2_ori_hist * c2_ori_area - c2_hist) / (c2_ori_area - len(c2_pixels))
        c2_hist = c2_hist / len(c2_pixels)
        return c1_hist, c1_exr_hist, c1_ori_hist, c1_pixels, c2_hist, c2_exr_hist, c2_ori_hist, c2_pixels

    def Int(self, mvBlock, source, distance):
        diff_sor = 0
        diff_dist = 0
        for i in range(513):
            if mvBlock[i] < source[i]:
                diff_sor += mvBlock[i]
            else:
                diff_sor += source[i]
            if mvBlock[i] < distance[i]:
                diff_dist += mvBlock[i]
            else:
                diff_dist += distance[i]

        if diff_sor < 0.1 and diff_dist > diff_sor:
            return True
        else:
            return False