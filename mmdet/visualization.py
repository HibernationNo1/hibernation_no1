import numpy as np
import cv2
import random
import math

import matplotlib.pyplot as plt


def mask_to_polygon(masks):
    polygons = []
    for mask in masks:       
        polygon, _ = bitmap_to_polygon(mask)
        if len(polygon) == 0:
            polygons.append([])
        else:
            polygons.append(polygon[0])
    return polygons


  
def bitmap_to_polygon(bitmap):
    """Convert masks from the form of bitmaps to polygons.

    Args:
        bitmap (ndarray): masks in bitmap representation.

    Return:
        list[ndarray]: the converted mask in polygon representation.
        bool: whether the mask has holes.
    """
    bitmap = np.ascontiguousarray(bitmap).astype(np.uint8)
    # cv2.RETR_CCOMP: retrieves all of the contours and organizes them
    #   into a two-level hierarchy. At the top level, there are external
    #   boundaries of the components. At the second level, there are
    #   boundaries of the holes. If there is another contour inside a hole
    #   of a connected component, it is still put at the top level.
    # cv2.CHAIN_APPROX_NONE: stores absolutely all the contour points.
    outs = cv2.findContours(bitmap, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = outs[-2]
    hierarchy = outs[-1]
    if hierarchy is None:
        return [], False
    # hierarchy[i]: 4 elements, for the indexes of next, previous,
    # parent, or nested contours. If there is no corresponding contour,
    # it will be -1.
    with_hole = (hierarchy.reshape(-1, 4)[:, 3] >= 0).any()
    contours = [c.reshape(-1, 2) for c in contours]
    return contours, with_hole


def draw_to_img(img, bboxes, labels, masks,
                class_names,
                score_thr=0.3): 
    """Draw `result` over `img`.

    Args:
        img (str | ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        mask (ndarray | None): Masks, shaped (n,h,w) or None.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown. Default: 0.

    """
    assert bboxes is None or bboxes.ndim == 2, \
        f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
    assert labels.ndim == 1, \
        f' labels ndim should be 1, but its ndim is {labels.ndim}.'
    assert bboxes is None or bboxes.shape[1] == 4 or bboxes.shape[1] == 5, \
        f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'
    assert bboxes is None or bboxes.shape[0] <= labels.shape[0], \
        'labels.shape[0] should not be less than bboxes.shape[0].'
    assert masks is None or masks.shape[0] == labels.shape[0], \
        'masks.shape[0] and labels.shape[0] should have the same length.'
    assert masks is not None or bboxes is not None, \
        'masks and bboxes should not be None at the same time.'    
    
    if score_thr > 0:
        assert bboxes is not None and bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        if masks is not None:
            masks = masks[inds, ...]
            
    
    scores = bboxes[:, -1]      # [num_instance]
    bboxes = bboxes[:, :4]      # [num_instance, [x_min, y_min, x_max, y_max]]
    
    object_labels = []
    for label_num, score in zip(labels, scores):
        confidence = int(round(score, 2)*100)
        object_name = class_names[label_num]
        object_labels.append(f"{object_name} {confidence}%")
    # object_labels len: num_instance,  each type: str,     e.g. object 34%

    # len(polygons) : num_instance,     len(polygons[n]): num_points of polygon
    polygons = mask_to_polygon(masks)
    
    org_img = img.astype(np.uint8)
    sub_img = np.zeros_like(org_img, dtype= np.uint8)   
    img_dict = dict(box = sub_img.copy(),
                    polygon = sub_img.copy(),
                    mask = sub_img.copy())
    
    colors = get_colors(len(object_labels))
    img_dict['box'] = draw_box(img_dict['box'], bboxes, colors) 
    img_dict['polygon'] = draw_polygon(img_dict['polygon'], polygons, colors)
    img_dict['mask'] = draw_mask(img_dict['mask'], polygons, colors)    
    
    return draw_by_projecting(org_img, img_dict, object_labels, bboxes, colors)


def draw_by_projecting(org_img, img_dict, object_labels, bboxes, colors):
    for key in img_dict.keys():
        if key == 'box': beta_1, beta_2 = 0.35, 0.5
        elif key == 'text':  
            org_img[img_dict[key] !=0] = 0
            continue
        elif key == "polygon":  beta_1, beta_2 = 0.4, 0.4
        elif key == "mask":  beta_1, beta_2 = 0.3, 0.4
        
        
        tmp_img = org_img.copy()
        tmp_img[img_dict[key] ==0] = 0
        org_img[img_dict[key] !=0] = 0
        
        org_img = cv2.addWeighted(src1 = org_img, alpha = 1.0, src2 = tmp_img, beta=beta_1, gamma = 0)
        org_img = cv2.addWeighted(src1 = org_img, alpha = 1.0, src2 = img_dict[key], beta=beta_2, gamma = 0)

    org_img = put_text(org_img, object_labels, bboxes, colors)
    return org_img

def get_colors(num_instance):
    colors = []
    for _ in range(num_instance):
        rgb = []
        for _ in range(3):
            rgb.append(random.randrange(100,255))
        colors.append(tuple(rgb))
    
    return colors

def draw_mask(img, polygons, colors):
    for polygon, color in zip(polygons, colors):
        cv2.fillPoly(img, [polygon], color = color)            
    return img


def draw_polygon(img, polygons, colors):
    for polygon, color in zip(polygons, colors):
        cv2.polylines(img, 
                        [polygon], 
                        isClosed = True, 
                        color = color, 
                        thickness = 2)
            
    return img
    

def draw_box(img, bboxes, colors, thickness= 2, lineType= cv2.LINE_4):
    for bbox, edge_color in zip(bboxes, colors):
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(img, 
                    (int(x_min), int(y_min)), (int(x_max), int(y_max)),
                    edge_color, 
                    lineType = lineType, 
                    thickness = thickness)
    return img


def put_text(img, object_labels, bboxes, colors, fontface = cv2.FONT_HERSHEY_SIMPLEX):
    for label_text, bboxe, color in zip(object_labels, bboxes, colors):
        x_min, y_min, x_max, y_max = bboxe
        width, height = x_max - x_min, y_max - y_min
        text_lb = (int(x_min), int(y_min))

        fontscale = max(math.sqrt(width*height)//70, 1)/3
        if fontscale >=1 : thickness = 2
        else: thickness = 1
        
        text_size, _ = cv2.getTextSize(label_text, fontFace = fontface, fontScale = fontscale, thickness = thickness)
        text_w, text_h = text_size
        text_rt = (int(x_min+text_w), int(y_min-text_h))
        
        cv2.rectangle(img, text_lb, text_rt, (0, 0, 0), thickness=-1)
        cv2.putText(img, label_text, 
                    org = text_lb, 
                    fontFace = fontface, 
                    fontScale = fontscale, 
                    color = color, 
                    lineType = cv2.LINE_8, 
                    thickness = thickness)
    return img


def draw_PR_curve(output_path, class_name, PR_list, ap_area, dv_flag = False, show_plot = False):
    precision_list, recall_list = [], []
    fig, ax = plt.subplots(figsize = (10, 5))
    for PR in PR_list:
        precision, recall, _ = PR
        precision_list.append(precision)
        recall_list.append(recall)
    
    ax.plot(recall_list, precision_list)
    fig.tight_layout(pad = 3)
    if dv_flag:
        ax.set_title(f"dv_PR of '{class_name}',  dv_ap_area : {ap_area:.4f}", fontsize = 20)
    else:
        ax.set_title(f"PR of '{class_name}',  ap_area : {ap_area:.4f}", fontsize = 20)

    ax.grid(axis = 'x', linewidth = 1, linestyle = '--')
    ax.grid(axis = 'y', linewidth = 1, linestyle = '--')
    ticks = np.arange(0, 1.1, 0.1)
    ax.set_xticks(ticks)  
    ax.set_yticks(ticks)  
    
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])

    plt.savefig(output_path)
    plt.close()
    if show_plot:
        plt.show()