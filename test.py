import cv2
import numpy as np
import os
import csv
from fixslic import fixslic_process
import matplotlib as plt
from basic_elements import Frame
from drawcontours import mergeClusters

def readLabel(labelPath):
    with open(labelPath, "r") as f:
        reader = csv.reader(f)
        label = list(reader)  # label = [['0', '0', '0', '0' ....
        for i in range(np.array(label).shape[0]):
            label[i] = list(map(int, label[i]))
        return label


if __name__ == '__main__':
    root = "C:\\Users\\PYN\\Desktop\\xueweilunwen\\MyProject"
    labelRoot = os.path.join(root, "output", "desk", "label")
    imgRoot = os.path.join(root, "output", "desk", "data")
    # imgName = ['00136.jpg', '00001.jpg']
    # labelName = ['slic_c_2000136.csv', 'slic_c_2000001.csv']
    imgName = ['00136.jpg', '00001.jpg']
    labelName = ['slic_c_2000136.csv', 'slic_c_2000001.csv']
    frames = []
    for i in range(2):
        imgPath = os.path.join(imgRoot,imgName[i])
        labelPath = os.path.join(labelRoot,labelName[i])
        img = cv2.imread("C:\\Users\\PYN\\Desktop\\xueweilunwen\\MyProject\\data\\images\\val\\42049.jpg")
        labels = readLabel("C:\\Users\\PYN\\Desktop\\xueweilunwen\\MyProject\\output\\label\\slic42049.csv")
        frame = Frame(img, labels)
        #边界上两点Pa、Pb的HSI欧式距离小于threshold1则Pa、Pb支持clusterA、clusterB融合；（a,b）边上支持merge的点/总点数>threshold2则clusterA、clusterB可融合

        # fix = fixslic_process(frame)
        # # fix.updatelabels_245(10)
        # frame = fix.updatelabels_process(4, 8)
        # fix = fixslic_process(frame)
        # frame = fix.updatelabels_process(4, 4)
        frame.isEdgeSupportMerge(5)
        # # input()
        frame.sortedges(10, 0.4)

        cv2.namedWindow("sort_point")
        cv2.imshow("sort_point", frame.img_sort_point)
        def Mousecallback(event, x, y, flags, param):
            if event == cv2.EVENT_FLAG_LBUTTON:
                print((x, y), frame.img_hsi[y, x])
        cv2.setMouseCallback("sort_point", Mousecallback)
        if cv2.waitKey(0) == 'q':
            pass

        merge_plan = frame.checkMergePlan()
        cv2.imshow("plan", merge_plan)
        frame.calculateLTY()

        frame.mergeClusters()
        # merge_result = frame.drawBlocks()
        # cv2.imshow("merge_result", merge_result)
        # cv2.waitKey(0)

        cv2.waitKey(0)
        frames.append(frame)


    print("initial done!")
    new_img = np.concatenate((frames[0].img_contours, frames[1].img_contours),axis=1)#合并两张图

    merge_img = mergeClusters(frames[0], 5, 99999)#找到是否可合并
    cv2.imshow("merge",merge_img)
    cv2.imwrite("output\\desk\\merge\\00001.jpg",merge_img)

    cv2.waitKey(0)
