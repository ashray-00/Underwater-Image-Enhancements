import numpy as np

class Node(object):
	def __init__(self,x,y,value):
		self.x = x
		self.y = y
		self.value = value
	def printInfo(self):
		print(self.x,self.y,self.value)

def getAtomsphericLight(darkChannel, img, meanMode, percent):
    size = darkChannel.shape[0] * darkChannel.shape[1]
    height = darkChannel.shape[0]
    width = darkChannel.shape[1]
    nodes = []
    img = np.float16(img)
    for i in range(0, height):
        for j in range(0, width):
            oneNode = Node(i, j, darkChannel[i, j])
            nodes.append(oneNode)
    nodes = sorted(nodes, key=lambda node: node.value, reverse=True)
    atomsphericLight = 0

    for i in range(0, int(percent*size)):
        SumImg = sum(img[nodes[i].x, nodes[i].y,:])
        if SumImg > atomsphericLight:
            atomsphericLight = SumImg
            AtomsphericLight = img[nodes[i].x, nodes[i].y, :]
            
    return AtomsphericLight

