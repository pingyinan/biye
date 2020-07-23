import cv2
import numpy as np
import queue
import heapq
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
        self.f_bin = ((img_hsi[:, :, 2]/32) + (img_hsi[:, :, 1]/32) * 8 + (img_hsi[:, :, 0]/32) * 8 * 8).astype(np.int)
        self.f_bin[self.f_bin > 512] = 512
        self.intial_clusters()
        self.findEdges()

    def updatesp(self):
        sp_area = np.zeros((1, self.sp_number), dtype=np.int)
        sp_position = np.zeros((self.sp_number, 2), dtype=int)  # [[y(0<y<height)],[x(0<x<width)]]
        sp_hist = np.zeros((self.sp_number, 513), dtype=float)
        for i in range(self.img.shape[0]):
            for j in range(self.img.shape[1]):
                label = self.labels[i][j]
                sp_area[0, label] += 1
                sp_position[label][0] += i
                sp_position[label][1] += j
                sp_hist[label][int(self.f_bin[i][j])] += 1
        for i in range(self.sp_number):
            area = sp_area[0, i]
            sp_position[i, :] = sp_position[i, :]/area
            sp_hist[i, :] = sp_hist[i, :]/area
        self.sp_area = sp_area
        self.sp_position = sp_position
        self.sp_hist = sp_hist

#初始化cluster，并更新其连通度，需要在findEdges之后
    def intial_clusters(self):
        self.updatesp()
        self.clusters = []
        for i in range(self.sp_number):
            self.clusters.append(self.make_cluster(i, (self.sp_position[i][0], self.sp_position[i][1]), self.sp_hist[i]))

    def make_cluster(self, index, pos, hist):
        return Cluster(index, pos, hist)

#找到所有cluster之间的边,并投票是否支持合并
    # def findEdges(self, threshold1):
    #     self.edges = {}
    #     self.isEdgeSupportMerge = {}   #投票支持聚合的点的比例
    #     img_hsi = self.img_hsi
    #     img_hsi = np.array(img_hsi, dtype=float)
    #     self.img_sort_point = self.img.copy()
    #     height, width, channel = self.img.shape
    #     dx = [-1, -1, 0, 1, 1, 1, 0, -1]
    #     dy = [0, -1, -1, -1, 0, 1, 1, 1]
    #     for j in range(height):
    #         for k in range(width):
    #             self.clusters[self.labels[j][k]].pixels.append((j, k))
    #             for i in range(8):
    #                 x = k + dx[i]
    #                 y = j + dy[i]
    #                 if x > 0 and x < width and y > 0 and y < height:
    #                     if self.labels[j][k] != self.labels[y][x]:
    #                         self.img_sort_point[j, k] = [0, 0, 255]
    #                         c1 = min(self.labels[j][k], self.labels[y][x])
    #                         c2 = max(self.labels[j][k], self.labels[y][x])
    #                         self.edges.setdefault((c1, c2), []).append((j, k))
    #                         a = img_hsi[j][k] - img_hsi[y][x]
    #                         # dist = np.linalg.norm(a)
    #                         dist = np.sum(np.maximum(a, -a))
    #                         if dist < threshold1:
    #                             self.img_sort_point[j, k] = [0, 255, 0]
    #                             if (c1, c2) in self.isEdgeSupportMerge:
    #                                 self.isEdgeSupportMerge[(c1, c2)] += 1.0
    #                             else:
    #                                 self.isEdgeSupportMerge[(c1, c2)] = 1.0
    #     for edge in self.edges:
    #         self.edges[edge] = list(set(self.edges[edge]))

    def findEdges(self):
        self.edges = {}
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
        for edge in self.edges:
            self.edges[edge] = list(set(self.edges[edge]))

    def isEdgeSupportMerge(self, threshold1):
        self.isEdgeSupportMerge = {}   #投票支持聚合的点的比例
        img_hsi = self.img_hsi
        img_hsi = np.array(img_hsi, dtype=float)
        self.img_sort_point = self.img.copy()
        height, width, channel = self.img.shape
        # dx = [-1, -1, 0, 1, 1, 1, 0, -1]
        # dy = [0, -1, -1, -1, 0, 1, 1, 1]
        dx = [-2, -2, 0, 2, 2, 2, 0, -2]
        dy = [0, -2, -2, -2, 0, 2, 2, 2]
        for edge, edge_points in self.edges.items():
            for point in edge_points:
                h, w = point
                self.img_sort_point[h, w] = [0, 0, 255]
                for i in range(8):
                    x = w + dx[i]
                    y = h + dy[i]
                    if x > 0 and x < width and y > 0 and y < height:
                        if (self.labels[h][w] == edge[0] and self.labels[y][x] == edge[1]) or (self.labels[h][w] == edge[1] and self.labels[y][x] == edge[0]):
                            a = img_hsi[h][w] - img_hsi[y][x]
                            dist = np.linalg.norm(a)
                            # dist = np.sum(np.maximum(a, -a))
                            if dist < threshold1:
                                self.img_sort_point[h, w] = [0, 255, 0]
                                if edge in self.isEdgeSupportMerge:
                                    self.isEdgeSupportMerge[edge] += 1.0
                                else:
                                    self.isEdgeSupportMerge[edge] = 1.0
                                break
        return 0

    def sortedges(self, threshold2, threshold3):
        for edge, value in self.isEdgeSupportMerge.items():    #注意有些边上没有任何点支持合并则不在self.isEdgeSupportMerge里
            length = len(self.edges[edge])
            if len(self.edges[edge]) < threshold2:   #边太短则不予考虑
                self.isEdgeSupportMerge[edge] = False
            else:
                if value / length > threshold3:
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

    def update(self,threshold1):
        self.intial_clusters()
        self.findEdges(threshold1)



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
        return img_merge

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
                                if self.mergeStrategy2(newMerge, clusterUnderCheck, edge):#判断可以聚合（hist是否同意聚合）
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
        self.choosen = []
        def Mousecallback(event, x, y, flags, param):
            color = ['red', 'gold', 'darkgreen', 'blue','gray','darksalmon','olivedrab',
                     'lightseagreen','darkorchid','navy','m','rosybrown','firebrick',
                     'chartreuse','royalblue','plum','silver']
            scale = np.arange(513)
            if event == cv2.EVENT_FLAG_LBUTTON:
                label = self.labels[y][x]
                cluster = self.clusters[label]
                self.choosen.append(cluster)
                print("label:", cluster.index)
                print("edges:", cluster.edges)
                print("liantongshu" ,cluster.liantongshu)
                plt.figure()
                plt.title("label:{}".format(cluster.index))
                for i in range(len(self.choosen)):
                    hsi_hist = self.choosen[i].hsi_hist
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
            if event == cv2.EVENT_FLAG_RBUTTON:
                self.choosen = []
        cv2.namedWindow("merge process")
        cv2.setMouseCallback("merge process", Mousecallback)
        cv2.imshow("merge process", draw_img)
        cv2.waitKey(10)


#判断以edge想邻的cluster和merge是否可以合并
    def mergeStrategy(self, Merge, cluster, edge):
        c1, c2 = edge
        neighbor = int()
        if c1 == cluster.index:
            neighbor = c2
        else:
            neighbor = c1
        clusterB = self.clusters[neighbor]
        maxindexB = np.argmax(clusterB.hsi_hist, axis=0)
        maxindexA = np.argmax(cluster.hsi_hist, axis=0)
        if abs(maxindexA - maxindexB) < 32:
            return True
        else:
            return False
        # return True

    def mergeStrategy2(self, Merge, clusterUnderCheck, edge):
        c1, c2 = edge
        if (c1 == 209 and c2 == 249) or (c1 == 249 and c2 == 209):
            print(c1, c2)
        if c1 == clusterUnderCheck.index:
            expandpoint = c2
        else:
            expandpoint = c1
        expandCluster = self.clusters[expandpoint]
        scale = 16
        point_num = 5
        def cmpTwocluster(scale, point_num, expandCluster, clusterUnderCheck):
            max_index_list = heapq.nlargest(len(expandCluster.hsi_hist), range(len(expandCluster.hsi_hist)), expandCluster.hsi_hist.take)
            point = []
            k = 0
            while point_num:
                mid = max_index_list[k]
                if len(point) == 0:
                    point.append(mid)
                else:
                    if expandCluster.hsi_hist[mid] < 0.1 * expandCluster.hsi_hist[point[0]]:
                        break
                    is_choose = True
                    for cp in point:
                        if abs(mid - cp) < scale:
                            is_choose = False
                            break
                    if is_choose:
                        point.append(mid)
                        point_num -= 1
                k += 1
            choosen_bin = []
            for cp in point:
                start = cp - scale
                end = cp + scale + 1
                if start < 0:
                    start = 0
                if end > 513:
                    end = 513
                choosen_bin = choosen_bin + [n for n in range(start,end)]
            choosen_bin = list(set(choosen_bin))
            den = np.sum(expandCluster.hsi_hist[choosen_bin])

            # hist_diff = 0
            # temp = expandCluster.hsi_hist[choosen_bin] - clusterUnderCheck.hsi_hist[choosen_bin]
            # for t in temp:
            #     hist_diff += abs(t)

            hist_diff = abs(np.sum(expandCluster.hsi_hist[choosen_bin] - clusterUnderCheck.hsi_hist[choosen_bin]))

            result = hist_diff/den
            if result < 0.6:
                return True
            else:
                return False
        if cmpTwocluster(scale, point_num, expandCluster, clusterUnderCheck) and cmpTwocluster(scale, point_num, clusterUnderCheck, expandCluster):
            return True
        else:
            return False

        # for i in range(point_num):
        #     mid = max_index_list[i]
        #     start = mid - scale
        #     end = mid + scale
        #     if start < 0:
        #         start = 0
        #     if end > 513:
        #         end = 513
        #     point = point + [n for n in range(start,end)]
        # choosen_bin = list(set(point))
        # den = np.sum(expandCluster.hsi_hist[choosen_bin])
        # hist_diff = abs(np.sum(expandCluster.hsi_hist[choosen_bin] - clusterUnderCheck.hsi_hist[choosen_bin]))
        # result = hist_diff/den
        # if result < 0.3:
        #     return True
        # else:
        #     return False














