import os
import cv2
import xml.etree.ElementTree as ET
from lxml import etree, objectify
import numpy as np
import matplotlib.pyplot as plt


def _bbox_area_computer(bbox):
    width = bbox[1] - bbox[0]
    height = bbox[3] - bbox[2]
    return width * height


def read_annotation_xml(xml_file):
    tree = ET.parse(xml_file)

    imageid = int((tree.find('filename').text).replace('.jpg', ''))

    size = tree.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    image_size = [width, height]

    resolution = 0
    sensor = 0
    source_info = [resolution, sensor]
    objs = tree.findall('object')

    small_object = 0
    middle_object = 0
    large_object = 0

    bboxes = []

    for obj in objs:
        bndbox = obj.find('bndbox')
        [xmin, xmax, ymin, ymax] = [
            int(bndbox.find('xmin').text),
            int(bndbox.find('xmax').text),
            int(bndbox.find('ymin').text),
            int(bndbox.find('ymax').text)
        ]
        if xmin < 0:
            xmin = 0
        if ymin < 0:
            ymin = 0
        bbox = [xmin, xmax, ymin, ymax]
        bboxes.append(bbox)

        area_of_bbox = _bbox_area_computer(bbox)

        if area_of_bbox <= 32 * 32:
            small_object += 1
        elif area_of_bbox <= 64 * 64:
            middle_object += 1
        else:
            large_object += 1

    print(' Number of small objects:{0}'.format(small_object))
    print(' Number of middle objects:{0}'.format(middle_object))
    print(' Number of large objects:{0}'.format(large_object))
    return imageid, image_size, source_info, bboxes


def two_points_belong_to_the_orignial_bbox_area(bbox, temp_bbox):
    if (temp_bbox[0] - bbox[0]) * (temp_bbox[0] - bbox[1]) <= 0 and (temp_bbox[2] - bbox[2]) * (
            temp_bbox[2] - bbox[3]) <= 0 and (temp_bbox[1] - bbox[0]) * (temp_bbox[1] - bbox[1]) <= 0 and (
            temp_bbox[3] - bbox[2]) * (temp_bbox[3] - bbox[3]) <= 0:
        return True
    else:
        return False

def do_filter_to_inappropriate_data(subImage, threshold=16384): #if there are too many zero pixel in the image then return True, you can adjust the threshold to what you want
    zero_pixel_num = (subImage.reshape(-1,1)[:,0]==0).sum()
    if zero_pixel_num >= threshold:
        return True


def add_gt(img_data, bboxes):
    colors = (0, 0, 255)
    for bbox in bboxes:
        cv2.rectangle(img_data, (bbox[0], bbox[2]), (bbox[1], bbox[3]), colors, 5)
    return img_data


def tiff_image_visualization(img_data):
    img_data = 255 * (np.log2(img_data + 1) / 16)
    img_data = img_data.astype(np.uint8)
    return img_data

def save_to_xml(xml_path, Image_shape, source_info, sub_bboxes, img_name):
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.annotation(
        E.folder('SAR_ship_patch'),
        E.filename(img_name),
        E.source(
            E.database('UNKnow'),
            E.resolution(source_info[0]),
            E.sensor(source_info[1]),
        ),
        E.size(
            E.width(Image_shape[1]),
            E.height(Image_shape[0]),
            E.depth(Image_shape[2])
        ),
        E.segmented(0),
    )

    for bbox in sub_bboxes:
        E2 = objectify.ElementMaker(annotate=False)
        anno_tree2 = E2.object(
            E.name('ship'),
            E.bndbox(
                E.xmin(bbox[0]),
                E.ymin(bbox[2]),
                E.xmax(bbox[1]),
                E.ymax(bbox[3])
            ),
            E.difficult(0)
            )
        anno_tree.append(anno_tree2)

    etree.ElementTree(anno_tree).write(xml_path, pretty_print=True)


def crop_single_image(img_filename, width, height, bboxes,overlap_size_of_width,overlap_size_of_height,aimed_width,aimed_height,source_info,new_imageid):

    #read image,now the data is read as 16bit
    img_data = cv2.imread(os.path.join(raw_images_dir, img_filename),-1)
    #add one channel so it will be 3 channel
    #img_data = img_data[:,:,np.newaxis]
    img_data_show = add_gt(img_data.copy(), bboxes)
    if len(bboxes) > 0:# if there are object in the raw image
        shape = img_data.shape  #return in the format [h,w,c]
        print(shape)
        if width !=shape[1] or height !=shape[0]: #the width and height is the read one from the xml file and the shape is the actual one of the image
            print('Actual size of image do not equal the annotated one')
        else:
            font = cv2.FONT_HERSHEY_SIMPLEX
            countimage = 0
            for start_w in range(0, shape[1], overlap_size_of_width):
                # we use sliding window to crop the image and make decision about whether to save the image area
                # through the location relation between the curent area and all the bboxes
                # as well as the characteristic of the image area itself
                for start_h in range(0, shape[0], overlap_size_of_height):
                    countimage += 1
                    sub_bboxes = []
                    start_w_new = start_w
                    start_h_new = start_h

                    # the crop range cannot beyond the image itself
                    # if they do, then we sacrifice the overlap size in width and height
                    if start_w + aimed_width > shape[1]:
                        start_w_new = shape[1] - aimed_width
                    if start_h + aimed_height > shape[0]:
                        start_h_new = shape[0] - aimed_height
                    top_left_x_in_raw_image = max(start_w_new, 0)
                    top_left_y_in_raw_image = max(start_h_new, 0)
                    bottom_right_x_in_raw_image = min(start_w + aimed_width, shape[1])
                    bottom_right_y_in_raw_image = min(start_h + aimed_height, shape[0])

                    #crop image
                    subImage = img_data[top_left_y_in_raw_image : bottom_right_y_in_raw_image, top_left_x_in_raw_image : bottom_right_x_in_raw_image]

                    #we do this to filter the image with too many zero pixel value
                    if do_filter_to_inappropriate_data(subImage):
                        continue


                    for bbox in bboxes:
                        # at first, we calculate the location of the overlap area between the current image area and all the bboxes
                        # actually, if one bbox do not overlap with the current area, the lacation calculated will be different with the real overlapped one
                        # the calculated location will not belong to the area of the bbox itself
                        temp_top_left_point_x = max(bbox[0], top_left_x_in_raw_image) #xmin
                        temp_top_left_point_y = max(bbox[2], top_left_y_in_raw_image) #ymin
                        temp_bottom_right_point_x = min(bbox[1], bottom_right_x_in_raw_image)  #xmax
                        temp_bottom_right_point_y = min(bbox[3], bottom_right_y_in_raw_image) #ymax
                        temp_bbox = [temp_top_left_point_x, temp_bottom_right_point_x, temp_top_left_point_y, temp_bottom_right_point_y]

                        if two_points_belong_to_the_orignial_bbox_area(bbox, temp_bbox):
                            #make sure the bbox do overlap with the current image area
                            #then we need to choose whether to save it to the annotation file of the current image according to the overlap rate
                            #here we set the threshold as 0.7
                            orignial_area = _bbox_area_computer(bbox)
                            temp_area = _bbox_area_computer(temp_bbox)
                            overlap_rate = temp_area / orignial_area
                            print('No:{0} overlap rate : {1}'.format(countimage,overlap_rate))

                            if overlap_rate >= 0.7:
                                # in this part we first do some preparation work about visualization
                                # so you can see whether you crop it properly during this process
                                img_data_show = cv2.rectangle(img_data_show,
                                                                  (top_left_x_in_raw_image, top_left_y_in_raw_image), (
                                                                      bottom_right_x_in_raw_image,
                                                                      bottom_right_y_in_raw_image),
                                                                  (0, 0, 255), 10)
                                center_point_of_the_new_image = [0.5 * (top_left_x_in_raw_image + bottom_right_x_in_raw_image),
                                                    0.5 * (top_left_y_in_raw_image + bottom_right_y_in_raw_image)]

                                img_data_show = cv2.putText(img_data_show, '{}'.format(str(countimage)),
                                                                (int(center_point_of_the_new_image[0]), int(center_point_of_the_new_image[1])), cv2.FONT_HERSHEY_SIMPLEX, 5,
                                                                (0, 255, 255), 10)

                                # calculate the bbox location relative to the new image
                                new_top_left_point_x = temp_bbox[0] - top_left_x_in_raw_image
                                new_bottom_right_point_x = temp_bbox[1] - top_left_x_in_raw_image
                                new_top_left_point_y = temp_bbox[2] - top_left_y_in_raw_image
                                new_bottom_right_point_y = temp_bbox[3] - top_left_y_in_raw_image
                                new_bbox = [new_top_left_point_x,new_bottom_right_point_x,new_top_left_point_y,new_bottom_right_point_y]
                                sub_bboxes.append(new_bbox)

                        else:
                            continue

                    if sub_bboxes: #if the cropped area has object then save the image to the new dataset
                        new_imageid += 1
                        img_name = img_filename.replace('.jpg', '') + '_' + str(countimage) + '-' + str(
                            new_imageid) + '.jpg'
                        xml = os.path.join(xml_dir, img_name.replace('.jpg','.xml'))
                        save_to_xml(xml, subImage.shape, source_info,sub_bboxes, str(img_name))
                        img_path = os.path.join(img_dir,img_name)
                        cv2.imwrite(img_path, subImage)

            #plt.figure(img_filename)
            #plt.imshow(img_data_show, cmap=plt.cm.gray)
            #plt.show()

            print('----------------------------------------end----------------------------------------------')
    return new_imageid


raw_data = '/data/chenxr/sar/SAR-ship-patch/' # where you need to change
raw_images_dir = os.path.join(raw_data, 'images')
raw_label_dir = os.path.join(raw_data, 'annotations')
images = [i for i in os.listdir(raw_images_dir) if 'jpg' in i]
labels = [i for i in os.listdir(raw_label_dir) if 'xml' in i]
print('find image:', len(images))
print('find label:', len(labels))

overlap_size = [128, 128]  # [width,height]
aimed_size = [256, 256]  # [width,height]

save_dir = '/data/chenxr/sar/SAR-ship-patch/train/'
img_dir = os.path.join(save_dir, 'images')
xml_dir = os.path.join(save_dir, 'annotations')
if os.path.exists(img_dir) == False:
    os.mkdir(img_dir)
if os.path.exists(xml_dir) == False:
    os.mkdir(xml_dir)

print('the image will be crop into {0}*{1} with {2},{3} overlap in width and height...'.format(aimed_size[0],aimed_size[1],overlap_size[0],overlap_size[1]))

new_imageid = 0


for img in os.listdir(raw_images_dir):
    xml_file = os.path.join(raw_label_dir, img.replace('.jpg', '') + '.xml')
    imgid, image_size, source_info, bboxes = read_annotation_xml(xml_file)
    print(imgid)
    print(source_info)
    new_imageid = crop_single_image(img, image_size[0],image_size[1],bboxes,overlap_size[0],overlap_size[1],aimed_size[0],aimed_size[1],source_info,new_imageid)
