import xml.dom
import xml.dom.minidom
import os
# from PIL import Image
import cv2
import json
 
_IMAGE_PATH = r'/data/chenxr/sar/HRSID/HRSID_JPG/JPEGImages/'
 
_INDENT = ' ' * 8
_NEW_LINE = '\n'
_FOLDER_NODE = 'HRSID'
_ROOT_NODE = 'annotation'
_DATABASE_NAME = 'Unknown'
_ANNOTATION = 'VOC'
_AUTHOR = 'cxr'
_SEGMENTED = '0'
_DIFFICULT = '0'
_TRUNCATED = '0'
_POSE = 'Unspecified'
 
# _IMAGE_COPY_PATH= 'JPEGImages'
 
 
 
# 封装创建节点的过程
def createElementNode(doc, tag, attr):  # 创建一个元素节点
    element_node = doc.createElement(tag)
 
    # 创建一个文本节点
    text_node = doc.createTextNode(attr)
 
    # 将文本节点作为元素节点的子节点
    element_node.appendChild(text_node)
 
    return element_node
 
 
def createChildNode(doc, tag, attr, parent_node):
    child_node = createElementNode(doc, tag, attr)
 
    parent_node.appendChild(child_node)
 
 
# object节点比较特殊
def createObjectNode(doc, attrs):
    object_node = doc.createElement('object')
    print("创建object中")
    midname = "ship"
 
    createChildNode(doc, 'name', midname,
                    object_node)
    createChildNode(doc, 'pose',
                    _POSE, object_node)
    createChildNode(doc, 'truncated',
                    _TRUNCATED, object_node)
    createChildNode(doc, 'difficult',
                    _DIFFICULT, object_node)
    bndbox_node = doc.createElement('bndbox')
    # print("midname1[points]:",midname1["points"])
    createChildNode(doc, 'xmin', str(int(attrs[0])),
                    bndbox_node)
    createChildNode(doc, 'ymin', str(int(attrs[1])),
                    bndbox_node)
    createChildNode(doc, 'xmax', str(int(attrs[2]) + int(attrs[0])),
                    bndbox_node)
    createChildNode(doc, 'ymax', str(int(attrs[3]) + int(attrs[1])),
                    bndbox_node)
    object_node.appendChild(bndbox_node)
 
    return object_node
 
 
# 将documentElement写入XML文件
def writeXMLFile(doc, filename):
    tmpfile = open('tmp.xml', 'w')
    doc.writexml(tmpfile, addindent=' ' * 8, newl='\n', encoding='utf-8')
    tmpfile.close()
    # # 删除第一行默认添加的标记
    fin = open('tmp.xml')
    fout = open(filename, 'w')
    lines = fin.readlines()
    for line in lines[1:]:
        if line.split():
            fout.writelines(line)
    fin.close()
    fout.close()
 
 
if __name__ == "__main__":
    ##json文件路径和图片路径,
    json_path = r"/data/chenxr/sar/HRSID/HRSID_JPG/annotations/train_test2017.json"
    img_path = r"/data/chenxr/sar/HRSID/HRSID_JPG/JPEGImages/"
    Annotations_save_path = r'/data/chenxr/sar/HRSID/HRSID_JPG/annotations_xml/'
 
    fileList = os.listdir(img_path)
    # print(".....::")
    # print("fileList:", fileList)
    if fileList == 0:
        os._exit(-1)
        # 对于每一张图都生成对应的json文件
        # 对于每一张图都生成对应的json文件
    for imageName in fileList:
        saveName = imageName.strip(".jpg")
        print(imageName)
        # 得到xml文件的名字
        xml_file_name = os.path.join(Annotations_save_path, (saveName + '.xml'))
        img = cv2.imread(os.path.join(img_path, imageName))
        height, width, channel = img.shape
        my_dom = xml.dom.getDOMImplementation()
        doc = my_dom.createDocument(None, _ROOT_NODE, None)
        # 获得根节点
        root_node = doc.documentElement
        # folder节点
        createChildNode(doc, 'folder', _FOLDER_NODE, root_node)
        # filename节点
        createChildNode(doc, 'filename', saveName + '.jpg', root_node)
 
        # source节点
        source_node = doc.createElement('source')
        # source的子节点
        createChildNode(doc, 'database', _DATABASE_NAME, source_node)
        # createChildNode(doc, 'annotation', _ANNOTATION, source_node)
        # createChildNode(doc, 'image', 'flickr', source_node)
        root_node.appendChild(source_node)
        size_node = doc.createElement('size')
        createChildNode(doc, 'width', str(width), size_node)
        createChildNode(doc, 'height', str(height), size_node)
        createChildNode(doc, 'depth', str(channel), size_node)
        root_node.appendChild(size_node)
        # 创建segmented节点
        createChildNode(doc, 'segmented', _SEGMENTED, root_node)
        # print("创建object节点")
        ann_data = []
        # print(json_path1)
        with open(json_path, "r") as f:
            ann = json.load(f)
            # print(ann)
        for i in range(5603):        # 0 ~ 5603  寻找与jpg_image同名的列表
            # 从第一个filename 第一个id开始循环
            # i就是图片image_id
            filename = ann["images"][i]   # 字典
            file_name = filename["file_name"]
            if imageName == file_name:
                id = filename["id"]
                annotations = ann["annotations"]     # len(annotations) = 16951
                for j in range(len(annotations)):    # 0 - 16951
                    image_id = annotations[j]["image_id"]
                    if image_id == id:                      # 说明annotations中的该行信息属于该图片
                        annotations2 = annotations[j]              # 将image_id等于id的取出来
                        object_node = createObjectNode(doc, annotations2["bbox"])
                        root_node.appendChild(object_node)
                    else:
                        continue
                continue
        # 构建XML文件名称
        # 写入文件
        writeXMLFile(doc, xml_file_name)