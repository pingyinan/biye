import cv2
import numpy as np
import queue
import matplotlib.pyplot as plt
class Cluster:
    def __init__(self, index, pos, hist):
        self.edges = []
        self.pos = pos
        self.index = index
        self.hsi_hist = hist
        self.liantongshu = 0
        self.pixels = []
#从联通数最高的cluster开始聚合
class Merge:
    def __init__(self, index):
        self.index = index
        self.clusters = []
        self.edges = []
class Frame:
    def __init__(self, img, labels):
        N_superpixels = max(np.amax(np.array(labels), axis=1)) + 1  # The number of superpixels
        self.sp_number = N_superpixels
        self.img = img
        self.labels = labels

        img_hsi = self.rgb2hsi(img)
        self.img_hsi = img_hsi
        self.img_contours = self.DrawContoursAroundSegments()
        f_bin = img_hsi[:, :, 0]/32 + (img_hsi[:, :, 1]/32) * 8 + (img_hsi[:, :, 2]/32) * 8 * 8
        f_bin[f_bin > 512] = 512
        sp_area = np.zeros((1, N_superpixels), dtype=np.int)
        sp_position = np.zeros((N_superpixels, 2), dtype=int)  # [[y(0<y<height)],[x(0<x<width)]]
        sp_hist = np.zeros((N_superpixels, 513), dtype=float)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                label = labels[i][j]
                sp_area[0, label] += 1
                sp_position[label][0] += i
                sp_position[label][1] += j
                sp_hist[label][int(f_bin[i][j])] += 1
        for i in range(N_superpixels):
            area = sp_area[0, i]
            sp_position[i, :] = sp_position[i, :]/area
            sp_hist[i, :] = sp_hist[i, :]/area

        self.sp_area = sp_area
        self.sp_position = sp_position
        self.sp_hist = sp_hist

    def make_cluster(self, index, pos, hist):
        return Cluster(index, pos, hist)

#初始化cluster，并更新其连通度，需要在findEdges之后
    def intial_clusters(self):
        self.clusters = []
        for i in range(self.sp_number):
            self.clusters.append(self.make_cluster(i, (self.sp_position[i][0], self.sp_position[i][1]), self.sp_hist[i]))

#找到所有cluster之间的边,并投票是否支持合并
    def findEdges(self, threshold1, threshold2):
        self.edges = {}
        self.isEdgeSupportMerge = {}   #投票支持聚合的点的比例
        img_hsi = self.img_hsi
        img_hsi = np.array(img_hsi, dtype=float)
        height, width, channel = self.img.shape
        dx = [-1, -1, 0, 1, 1, 1, 0, -1]
        dy = [0, -1, -1, -1, 0, 1, 1, 1]
        for j in range(height):
            for k in range(width):
                self.clusters[self.labels[j][k]].pixels.append((j, k))
                for i in range(8):
                    x = k + dx[i]
                    y = j + dy[i]
                    if x > 0 and x < width and y > 0 and y < height:
                        if self.labels[j][k] != self.labels[y][x]:
                            c1 = min(self.labels[j][k], self.labels[y][x])
                            c2 = max(self.labels[j][k], self.labels[y][x])
                            self.edges.setdefault((c1, c2), []).append((j, k))
                            dist = np.linalg.norm(img_hsi[j][k] - img_hsi[y][x])
                            if dist < threshold1:
                                if (c1, c2) in self.isEdgeSupportMerge:
                                    self.isEdgeSupportMerge[(c1, c2)] += 1.0
                                else:
                                    self.isEdgeSupportMerge[(c1, c2)] = 1.0
        for edge, value in self.isEdgeSupportMerge.items():    #注意有些边上没有任何点支持合并则不在self.isEdgeSupportMerge里
            length = len(self.edges[edge])
            if len(self.edges[edge]) < 20:
                self.isEdgeSupportMerge[edge] = False
            else:
                if value / length > threshold2:
                    self.isEdgeSupportMerge[edge] = True
                else:
                    self.isEdgeSupportMerge[edge] = False
        for edge in self.edges.keys():
            if edge not in self.isEdgeSupportMerge.keys():
                self.isEdgeSupportMerge[edge] = False
            c1, c2 = edge
            self.clusters[c1].edges.append(edge)
            self.clusters[c2].edges.append(edge)



    def calculateLTY(self):
        for edge in self.isEdgeSupportMerge.keys():  #计算cluster的连通度
            if self.isEdgeSupportMerge[edge]:
                self.clusters[edge[0]].liantongshu += 1
                self.clusters[edge[1]].liantongshu += 1

    def update(self,threshold1, threshold2):
        self.intial_clusters()
        self.findEdges(threshold1, threshold2)
        self.calculateLTY()


    def drawEdges(self):
        img_countours = self.img.copy()
        for edge, points in self.edges.items():
            for point in points:
                img_countours[point[0], point[1]] = [0, 0, 0]
        return img_countours



    def checkMergePlan(self):
        img_merge = self.img.copy()
        for edge, value in self.isEdgeSupportMerge.items():
            if value:
                for point in self.edges[edge]:
                    h = point[0]
                    w = point[1]
                    img_merge[h, w] = [0, 255, 0]
            else:
                for point in self.edges[edge]:
                    h = point[0]
                    w = point[1]
                    img_merge[h, w] = [0, 0, 255]
        self.mergeplan = img_merge
        # def Mousecallback(event, x, y, flags, param):
        #     if event == cv2.EVENT_FLAG_LBUTTON:
        #         label = self.labels[y][x]
        #         cluster = self.clusters[label]
        #         print("label:", cluster.index)
        #         print("edges:", cluster.edges)
        #         print("liantongshu" ,cluster.liantongshu)
        #         hsi_hist = cluster.hsi_hist
        #         scale = np.arange(513)
        #         plt.figure()
        #         plt.title("label:{}".format(cluster.index))
        #         plt.plot(scale, hsi_hist)
        #         plt.show()
        # cv2.namedWindow("mergeplan")
        # cv2.setMouseCallback("mergeplan", Mousecallback)
        # cv2.imshow("mergeplan",img_merge)


    def printstatus(self):
        cv2.imshow("picture",self.img)
        print("****** The information of img ******")
        h, w, c = self.img.shape
        print("height:{0} weight:{1} channel:{2}" .format(h, w, c))
        print("Superpixels'number:",self.sp_number)
        if cv2.waitKey(0) == 'q':
            pass

    def rgb2hsi(self, rgb_img):
        height, width, channel = rgb_img.shape
        b, g, r = cv2.split(rgb_img)
        b = b / 255.0
        g = g / 255.0
        r = r / 255.0
        hsi_img = rgb_img.copy()
        # hsi_img = np.array(hsi_img, dtype=np.float)
        for i in range(height):
            for j in range(width):
                num = r[i][j] - 0.5 * (g[i][j] + b[i][j])
                den = np.sqrt((r[i][j] - g[i][j]) ** 2 + (r[i][j] - b[i][j]) * (g[i][j] - b[i][j]))
                if den == 0:
                    H = 0
                else:
                    theta = float(np.arccos(num / den))
                    if g[i][j] >= b[i][j]:
                        H = theta
                    else:
                        H = 2 * np.pi - theta
                H = H / (2 * np.pi)
                sum = r[i][j] + g[i][j] + b[i][j]
                if sum == 0:
                    S = 0
                else:
                    S = 1 - 3 * (min(min(r[i][j], g[i][j]), b[i][j])) / sum
                I = (r[i][j] + g[i][j] + b[i][j]) / 3.0
                hsi_img[i][j][0] = H*255
                hsi_img[i][j][1] = S*255
                hsi_img[i][j][2] = I*255
                # normally the range of H is [0,2*pi],S、I is [0,1],now we normalize them to [0,1] together
        return hsi_img

    def DrawContoursAroundSegments(self):
        dx = [-1, -1, 0, 1, 1, 1, 0, -1]
        dy = [0, -1, -1, -1, 0, 1, 1, 1]
        height, width, channel = self.img.shape
        img_countours = self.img.copy()
        for j in range(height):
            for k in range(width):
                np = 0
                for i in range(8):
                    x = k + dx[i]
                    y = j + dy[i]
                    if x > 0 and x < width and y > 0 and y < height:
                        if self.labels[j][k] != self.labels[y][x]:
                            np = np + 1
                if np > 1:
                    img_countours[j, k] = [0, 0, 0]
        return img_countours

    def mergeClusters(self):
        sortedClusters = sorted(self.clusters, key= lambda x:x.liantongshu, reverse=True)
        hasMerged = [-1]*self.sp_number
        self.blocks = []
        n_merge = -1
        for cluster in sortedClusters:
            if hasMerged[cluster.index] == -1:  #初始化腐蚀点，每一个区块的聚合从其中连通度最高的点开始向外腐蚀。结束后再找到未聚合的连通度最高点，此点必定在另一块未聚合区块
                n_merge += 1
                hasMerged[cluster.index] = n_merge
                newMerge = Merge(n_merge)
                newMerge.clusters.append(cluster)
                newMerge.edges = newMerge.edges + cluster.edges
                q = queue.Queue()
                q.put(cluster)
                while not q.empty(): #从一点开始向外腐蚀（深搜）
                    expandPoint = q.get()
                    for edge in expandPoint.edges:
                        if self.isEdgeSupportMerge[edge]:  #如果边支持聚合
                            neighbor = int()
                            c1, c2 = edge
                            if c1 == expandPoint.index:
                                neighbor = c2
                            else:
                                neighbor = c1
                            clusterUnderCheck = self.clusters[neighbor]
                            if hasMerged[neighbor] == -1: #没有聚合过则判断，聚合过则跳过
                                if self.mergeStrategy(newMerge, clusterUnderCheck, edge):#判断可以聚合（hist是否同意聚合）
                                    hasMerged[neighbor] = newMerge.index  #更改cluster的分组
                                    newMerge.clusters.append(clusterUnderCheck)  #将cluster加入分组
                                    for margin in clusterUnderCheck.edges:  #边有重复说明这条边被聚合了，则从merge的轮廓上去除
                                        if margin in newMerge.edges:
                                            newMerge.edges.remove(margin)
                                        else:
                                            newMerge.edges.append(margin)
                                    q.put(clusterUnderCheck)  #cluster成为继续向外腐蚀的点
                        self.drawMergeProcess(newMerge, expandPoint)
                self.blocks.append(newMerge)
        self.drawMergeResult()

    def drawMergeResult(self):
        draw_img = self.img.copy()
        for block in self.blocks:
            for edge in block.edges:
                for h, w in self.edges[edge]:
                    draw_img[h, w] = [255, 0, 0]
        cv2.imshow("merge result", draw_img)

    def drawMergeProcess(self, newMerge, expandPoint):
        draw_img = self.mergeplan.copy()
        list = self.blocks.copy()
        list.append(newMerge)
        for block in list:
            for edge in block.edges:
                for h, w in self.edges[edge]:
                    draw_img[h, w][0] = 255
        for edge in expandPoint.edges:
            for h, w in self.edges[edge]:
                draw_img[h, w][0] = 255
        def Mousecallback(event, x, y, flags, param):
            if event == cv2.EVENT_FLAG_LBUTTON:
                label = self.labels[y][x]
                cluster = self.clusters[label]
                print("label:", cluster.index)
                print("edges:", cluster.edges)
                print("liantongshu" ,cluster.liantongshu)
                hsi_hist = cluster.hsi_hist
                scale = np.arange(513)
                plt.figure()
                plt.title("label:{}".format(cluster.index))
                plt.plot(scale, hsi_hist)
                plt.show()
        cv2.namedWindow("merge process")
        cv2.setMouseCallback("merge process", Mousecallback)
        cv2.imshow("merge process", draw_img)
        cv2.waitKey(10)










#判断以edge想邻的cluster和merge是否可以合并
    def mergeStrategy(self, Merge, cluster, edge):
        return 1




    # def mergeClusters(self, threshold):
    #     self.clusterSortedByLiantongshu = {}
    #     self.merges = {}
    #     hasMerged = [False] * self.sp_number
    #     for cluster in self.clusters:
    #         N = cluster.liantongshu
    #         self.clusterSortedByLiantongshu.setdefault(N, []).append(cluster)
    #     q = queue.Queue()
    #     for num in sorted(self.clusterSortedByLiantongshu.keys(), reverse=True):
    #         q.put(cluster for cluster in self.clusterSortedByLiantongshu(num))
    #     while not q.empty():
    #         cluster = q.get()
    #         if not hasMerged[cluster.index]:
    #             hasMerged[cluster.index] = True
    #             for edge in cluster.edges:
    #                 if self.isEdgeSupportMerge[edge] > threshold:
    #                     neighbor = int()
    #                     c1, c2 = edge
    #                     if c1 == cluster.index:
    #                         neighbor = c2
    #                     else:
    #                         neighbor = c1











