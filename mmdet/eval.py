import numpy as np
from hibernation_no1.mmdet.inference import inference_detector, parse_inferece_result
from hibernation_no1.mmdet.visualization import mask_to_polygon

def compute_iou(infer_box, gt_box):
    """
    infer_box : x_min, y_min, x_max, y_max
    gt_box : x_min, y_min, x_max, y_max
    """
    box1_area = (infer_box[2] - infer_box[0]) * (infer_box[3] - infer_box[1])
    box2_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    
    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(infer_box[0], gt_box[0])       # max of x_min
    y1 = max(infer_box[1], gt_box[1])       # max of y_min 
    x2 = min(infer_box[2], gt_box[2])       # min of x_max 
    y2 = min(infer_box[3], gt_box[3])       # min of y_max 

    # compute the width and height of the intersection
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)

    inter = w * h
    outer = (box1_area + box2_area - inter)
    if outer == 0:
        return 1.0
    iou = inter / outer
    return iou


def get_divided_polygon(polygon, window_num):
    """
        divide polygon by the number of `window_num` piece by sort in x and y direction
    Args:
        polygon (list): 
        window_num (int): 
    
    Return 
        [x_lt_rb_list, y_lt_rb_list]
            len(x_lt_rb_list) == `window_num`
            x_lt_rb_list[0]: [x_min, y_min, x_max, y_max]
    """
    if isinstance(polygon, np.ndarray): polygon = polygon.tolist()
  
    piece_point = int(len(polygon)/window_num)
    polygon_xsort = polygon.copy()
    polygon_xsort.sort(key=lambda x: x[0])
    polygon_ysort = polygon.copy()
    polygon_ysort.sort(key=lambda x: x[1])
    
    
    xsort_div_pol = divide_polygon(polygon_xsort, window_num, piece_point)
    ysort_div_pol = divide_polygon(polygon_ysort, window_num, piece_point)

    x_lt_rb_list, y_lt_rb_list = [], []
    for x_pol, y_pol in zip(xsort_div_pol, ysort_div_pol):
        x_lt_rb_list.append(get_box_from_pol(x_pol))
        y_lt_rb_list.append(get_box_from_pol(y_pol))
    
    
    return [x_lt_rb_list, y_lt_rb_list]
    
    
 
def divide_polygon(polygon_sorted, window_num, piece_point):
    """divide polygon by the number of `window_num` piece

    Args:
        polygon_sorted (list): 
        window_num (int): 
        piece_point (int): 

    Returns:
        sort_list: list, len== [`window_num`],      window_num[n]: list
    """
    sort_list = []
    last_num = 0
    for i in range(window_num):
        if i == window_num-1:
            sort_list.append(polygon_sorted[last_num:])
            break
        
        sort_list.append(polygon_sorted[last_num:piece_point*(i+1)])
        last_num = piece_point*(i+1)
    
    return sort_list    
    

def get_box_from_pol(polygon):
    x_min, y_min, x_max, y_max = 100000, 100000, -1, -1
    
    for point in polygon:
        x, y = point[0], point[1]
        if x < x_min and x > x_max:
            x_min = x_max = x
        elif x < x_min:
            x_min = x
        elif x > x_max:
            x_max = x
        
        if y < y_min and y > y_max:
            y_min = y_max = y
        elif y < y_min:
            y_min = y
        elif y > y_max:
            y_max = y
    
    return [x_min, y_min, x_max, y_max]
        

                

class Evaluate():
    def __init__(self, model, cfg, dataloader):
        self.model = model
        self.cfg = cfg
        self.dataloader = dataloader
        self.classes = self.model.CLASSES
        self.confusion_matrix = dict()
        self.set_treshold()
        
        self.get_precision_recall_value()
        self.precision_recall_dict = self.confusion_matrix
   
    
    def set_treshold(self):
        num_thrshd_divi = self.cfg.num_thrs_divi
        thrshd_value = (self.cfg.iou_thrs[-1] - self.cfg.iou_thrs[0]) / num_thrshd_divi
        self.iou_threshold = [round(self.cfg.iou_thrs[0] + (thrshd_value*i), 2) for i in range(num_thrshd_divi+1)]
    
    def compute_F1_score(self):
        F1_score_dict = dict()
        for class_name, threshold_list in self.precision_recall_dict.items():
            print(f"\nclass_name : {class_name}")
            F1_score_dict[class_name] = []
            for idx, threshold in enumerate(threshold_list):
                F1_score_dict[class_name].append(threshold['F1_score'])
        
        return F1_score_dict
            
    
    def compute_mAP(self):  
        classes_AP = self.compute_PR_area()
        
        sum_AP = 0
        for class_name, AP in classes_AP.items():
            assert AP <=1.0
            sum_AP +=AP
        mAP = sum_AP/len(self.model.CLASSES)
        
        return round(mAP, 4) 
      
    def compute_PR_area(self):
        classes_AP = dict()
        for class_name, threshold_list in self.precision_recall_dict.items():
            prcs_rcl_list = []
            for idx, threshold in enumerate(threshold_list):
                prcs_rcl_list.append([threshold['precision'], threshold['recall']])
            prcs_rcl_list.reverse()
            
            before_recall = sum_recall = 0
            for idx, prcs_rcl in enumerate(prcs_rcl_list):
                _, recall = prcs_rcl
                sum_recall +=abs(recall - before_recall)
                before_recall = recall
                
            
            # for adjust sum of recall
            adjust_value = 1
            if sum_recall > 1:
                # ideally, the values of recall are sorted in ascending order
                # but there are cases where the value rises and then falls.
                #   (by compute iou by divied polygons)
                # In this case, `sum_recall` exceed 1 sometimes.
                # to adjust this `sum_recall`, we need `adjust_value`
                adjust_value  = 1/sum_recall  
            
            ap_area = 0
            before_recall =0
            continue_count = -1
            for idx, prcs_rcl in enumerate(prcs_rcl_list):
                if continue_count > 0: 
                    continue_count -=1 
                    if continue_count == 0 : continue_count = -1
                    continue
                precision, recall = prcs_rcl
                
                if idx+1 < len(prcs_rcl_list) and prcs_rcl_list[idx+1][1] == recall:
                    tmp_precision_list = []
                    for i in range(idx, len(prcs_rcl_list)):
                        if recall == prcs_rcl_list[i][1]:       # same recall
                            continue_count +=1
                            tmp_precision_list.append(prcs_rcl_list[i][0])
                        else: break
                    precision = max([precis for precis in tmp_precision_list])
                
                # print(f"before_recall : {before_recall:.2f}, recall : {recall:.2f}    {abs(before_recall - recall):.2f}
                # precision: {precision:.2f}     area = {abs(before_recall - recall)*precision:.2f}")
                area = abs(before_recall - recall)*adjust_value*precision
                ap_area += area
            
                
                before_recall = recall
                    
            classes_AP[class_name] = round(ap_area, 4)
        return classes_AP
    
    def compute_precision_recall(self):
        for class_name, threshold_list in self.confusion_matrix.items():
            for idx, threshold in enumerate(threshold_list):
                if threshold['num_pred'] == 0: precision = 0
                else: precision = threshold['num_true']/threshold['num_pred']
                
                if threshold['num_gt'] == 0:    # class: `class_nema` is not exist in dataset
                    pre_recall = recall = 0
                else:  
                    pre_recall = threshold['num_iou']/threshold['num_gt']
                    recall = threshold['num_true']/threshold['num_gt']
    
                self.confusion_matrix[class_name][idx]['pre_recall'] = pre_recall
                self.confusion_matrix[class_name][idx]['recall'] = recall
                self.confusion_matrix[class_name][idx]['precision'] = precision
            
                if recall == 0 and precision == 0: self.confusion_matrix[class_name][idx]['F1_score'] =0
                else: self.confusion_matrix[class_name][idx]['F1_score'] = 2*(precision*recall)/(precision+recall)
        

    def get_precision_recall_value(self):
        for class_name in self.classes:
            self.confusion_matrix[class_name] = []
            for i in range(len(self.iou_threshold)):
                self.confusion_matrix[class_name].append(dict(iou_threshold = self.iou_threshold[i],
                                                num_gt = 0, 
                                                num_pred = 0, 
                                                num_true = 0,
                                                num_iou = 0)
                                            )
        
        for i, val_data_batch in enumerate(self.dataloader):
            gt_bboxes_list = val_data_batch['gt_bboxes'].data
            gt_labels_list = val_data_batch['gt_labels'].data
            img_list = val_data_batch['img'].data
            gt_masks_list = val_data_batch['gt_masks'].data
            assert len(gt_bboxes_list) == 1 and (len(gt_bboxes_list) ==
                                                    len(gt_labels_list) ==
                                                    len(img_list) == 
                                                    len(gt_masks_list))
            # len: batch_size
            batch_gt_bboxes = gt_bboxes_list[0]           
            batch_gt_labels = gt_labels_list[0]  
            batch_gt_masks = gt_masks_list[0]    
            
            img_metas = val_data_batch['img_metas'].data[0]
            batch_images_path = []    
            for img_meta in img_metas:
                batch_images_path.append(img_meta['filename'])
            batch_results = inference_detector(self.model, batch_images_path, self.cfg.batch_size)
                        
            assert (len(batch_gt_bboxes) == 
                        len(batch_gt_labels) ==
                        len(batch_images_path) ==
                        len(batch_gt_masks) ==
                        len(batch_results))
            batch_conf_list = [batch_gt_masks, batch_gt_bboxes, batch_gt_labels, batch_results]
            
            self.get_confusion_value(batch_conf_list)
                                        
        self.compute_precision_recall()
    
    
    def get_confusion_value(self, batch_conf_list):
        batch_gt_masks, batch_gt_bboxes, batch_gt_labels, batch_results = batch_conf_list
        
        for gt_mask, gt_bboxes, gt_labels, result in zip(
            batch_gt_masks, batch_gt_bboxes, batch_gt_labels, batch_results
            ):
            
            i_bboxes, i_labels, i_mask = parse_inferece_result(result)

            if i_mask is not None:
                if self.iou_threshold[0] > 0:
                    assert i_bboxes is not None and i_bboxes.shape[1] == 5
                    scores = i_bboxes[:, -1]
                    inds = scores > self.iou_threshold[0]
                    i_bboxes = i_bboxes[inds, :]
                    i_labels = i_labels[inds]
                    if i_mask is not None:
                        i_mask = i_mask[inds, ...]
                
                i_cores = i_bboxes[:, -1]      # [num_instance]
                i_bboxes = i_bboxes[:, :4]      # [num_instance, [x_min, y_min, x_max, y_max]]

                
                i_polygons = mask_to_polygon(i_mask)
                gt_polygons = mask_to_polygon(gt_mask.masks)
            else:   # detected nothing
                i_cores = i_bboxes = i_polygons = gt_polygons = []
                
            infer_dict = dict(bboxes = i_bboxes,
                            polygons = i_polygons,
                            labels = i_labels,
                            score = i_cores)
            gt_dict = dict(bboxes = gt_bboxes,
                        polygons = gt_polygons,
                        labels = gt_labels)
            
            
            self.get_num_posi_nega(gt_dict, infer_dict)  
    
    
    
    def get_num_posi_nega(self, gt_dict, infer_dict):
        num_pred = len(infer_dict['bboxes'])                    
        num_gt = len(gt_dict['bboxes'])
        for idx, threshold in enumerate(self.iou_threshold):        
            done_gt = []        
            for i in range(num_pred):
                pred_class_name = self.classes[infer_dict['labels'][i]]
                for j in range(num_gt):
                    gt_class_name = self.classes[gt_dict['labels'][j]]
                    
                    # number of gt of each threshold by class_name
                    if i == 0: self.confusion_matrix[gt_class_name][idx]['num_gt'] +=1 
                                    
                    if j in done_gt: continue
                    
                    i_bboxes, gt_bboxes = infer_dict['bboxes'][i], gt_dict['bboxes'][j]
                    iou = compute_iou(i_bboxes, gt_bboxes)
                    
                    if (iou > threshold and          
                        infer_dict['score'][i] > self.cfg.confidence_thrs):
                        if pred_class_name == gt_class_name:  
                            self.confusion_matrix[pred_class_name][idx]['num_iou'] +=1
                        
                        # compute iou by sliced polygon 
                        i_polygons, gt_polygons = infer_dict['polygons'][i], gt_dict['polygons'][j]
                        i_xsort_bbox_list, i_ysort_bbox_list = get_divided_polygon(i_polygons, 3)
                        gt_xsort_bbox_list, gt_ysort_bbox_list = get_divided_polygon(gt_polygons, 3)
                    
                        for i_xsort_bbox, gt_xsort_bbox in zip(i_xsort_bbox_list, gt_xsort_bbox_list):
                            if compute_iou(i_xsort_bbox, gt_xsort_bbox) < threshold:  continue
                        for i_ysort_bbox, gt_ysort_bbox in zip(i_ysort_bbox_list, gt_ysort_bbox_list):
                            if compute_iou(i_ysort_bbox, gt_ysort_bbox) < threshold:  continue
                        
                        # for calculate recall by divided_polygon
                        self.confusion_matrix[pred_class_name][idx]['num_pred'] +=1
                        
                        if pred_class_name == gt_class_name: 
                            self.confusion_matrix[pred_class_name][idx]['num_true'] +=1
                        
                        done_gt.append(j)
 
