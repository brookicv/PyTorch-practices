import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np 

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


if __name__ == "__main__":
    blocks = parse_cfg("yolov3.cfg")
    for block in blocks:
        print(block)