import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
from util import predict_transform

def parse_cfg(cfgfile):
    """
    解析网络结构的cfg文件

    返回一个包含网络结构的字典列表，每个字典表示网络结构中的一个块。
    """
    # 读取cfg文件
    file = open(cfgfile,"r")
    lines = file.read().split("\n")
    lines = [x for x in lines if len(x) > 0] # empty lines
    lines = [x for x in lines if x[0] != "#"] # comments
    lines = [x.rstrip().lstrip() for x in lines] # whitespaces

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":  # 一个block的开始,也意味着上一个block的结束
            if len(block) != 0:
                blocks.append(block) # 上一个block解析完成，添加
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key,value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks


class EmptyLayer(nn.Module):
    """
    为shortcut layer / route layer准备，具体功能不在此实现，在模型的forward中
    """
    def __init__(self):
        super(EmptyLayer,self).__init__()

class DetectionLayer(nn.Module):
    """
    检测层的具体实现，在特征图上使用anchor预测目标区域和类别，
    此功能在predict_transform中实现
    """
    def __init__(self,anchors):
        super(DetectionLayer,self).__init__()
        self.anchors = anchors

def create_modules(blocks):
    net_info = blocks[0] # 第一个block是关于网络训练的一些配置信息

    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for index,x in enumerate(blocks[1:]):
        module = nn.Sequential()

        if (x["type"] == "convolutional"):
            activation = x["activation"] # 激活函数
            # 是否进行BN
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True 
            filters = int(x["filters"]) # 卷积个数
            padding = int(x["pad"]) # 是否进行padding
            kernel_size = int(x["size"]) # 卷积核大小
            stride = int(x["stride"]) # 卷积步长

            if padding:
                pad = (kernel_size - 1) // 2 # padding的宽度，保持feature map大小不变
            else:
                pad = 0
            
            # 创建卷积层
            conv = nn.Conv2d(prev_filters,filters,kernel_size,stride,
                pad,bias=bias)
            module.add_module("conv_{}".format(index),conv)

            # BN
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{}".format(index),bn)

            # activation
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1,inplace=True)
                module.add_module("leaky_{}".format(index),activn)
        
        # an upsampling layer
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor=2,mode="bilinear")
            module.add_module("upsample_{}".format(index),upsample)

        # a route layer
        # route layer的作用，当layer的取值为正时，输出这个正数对应层的特征
        # 当layer的取值为负的时候，输出route层向后退layer层对应的
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(",")

            start = int(x["layers"][0])

            try:
                end = int(x["layers"][1])
            except:
                end = 0

            if start > 0:
                start = start - index 
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{}".format(index),route)

            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        # shortcut layer
        elif (x["type"] == "shortcut"):
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index),shortcut)

        # yolo is the detection layer
        elif (x["type"] == "yolo"):
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i],anchors[i+1]) for i in range(0,len(anchors),2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detecion_{}".format(index),detection)
    
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (net_info,module_list)

class Darknet(nn.Module):
    def __init__(self,cfgfile):
        super(Darknet,self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info,self.module_list = create_modules(self.blocks)

    def forward(self,x,CUDA):
        modules = self.blocks[1:]
        outputs = {} 
        
        write = 0
        for i,module in enumerate(modules):
            module_type = (module["type"])

            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x) # forward
            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]

                if layers[0] > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + layers[0]]
                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] - i
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1,map2),1)
            elif module_type == "shortcut":
                form_ = int(module["from"])
                x = outputs[i-1] + outputs[i + form_] # 求和

            elif module_type == "yolo":
                
                anchors = self.module_list[i][0].anchors

                inp_dim = int(self.net_info["height"])
                num_classes = int(module["classes"])

                x = x.data # 得到yolo层的feature map
                x = predict_transform(x,inp_dim,anchors,num_classes,CUDA)

                if not write:
                    detections = x 
                    write = 1 
                else:
                    detections = torch.cat((detections,x),1)
            outputs[i] = x
        return detections

if __name__ == "__main__":
    blocks = parse_cfg("yolov3.cfg")
    print(create_modules(blocks))