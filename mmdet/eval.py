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
        polygon (list): [[x_1, y_1], [x_2, y_2], ....]
        window_num (int): 
    
    Return 
        [x_lt_rb_list, y_lt_rb_list]
            len(x_lt_rb_list) == `window_num`
            x_lt_rb_list[0]: [x_min, y_min, x_max, y_max]
    """
    if isinstance(polygon, np.ndarray): polygon = polygon.tolist()
  
    piece_point = int(len(polygon)/window_num)
  
    polygon_xsort = polygon.copy()
    polygon_xsort.sort(key=lambda x: x[0])      # sort points by x coordinates
    polygon_ysort = polygon.copy()
    polygon_ysort.sort(key=lambda x: x[1])      # sort points by y coordinates
    
    
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
        polygon_sorted (list): polygon point
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
    """
        compute each left_top, right_bottom point of bbox
    """
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
   
    
    def set_treshold(self):
        num_thrshd_divi = self.cfg.num_thrs_divi
        thrshd_value = (self.cfg.iou_thrs[-1] - self.cfg.iou_thrs[0]) / num_thrshd_divi
        self.iou_threshold = [round(self.cfg.iou_thrs[0] + (thrshd_value*i), 2) for i in range(num_thrshd_divi+1)]
    
    def compute_F1_score(self, threshold_idx = 7):
        """
            threshold_idx: index threshold list
                ex) threshold list = [0.3, 0.4, 0.5, 0.6, 0.7], and threshold_idx = 3
                    0.6 = self.cfg.iou_thrs[3]         
            
            return:
                F1_score['F1_score'] have each class_names key
                F1_score['F1_score']['some_class_name'] : F1_score value
                F1_score['dv_F1_score'] is same
        """
        assert len(self.cfg.iou_thrs) > threshold_idx
        
        F1_score = dict(F1_score = dict(),
                        dv_F1_score = dict())
        for class_name, threshold_list in self.precision_recall_dict.items():
            F1_score['F1_score'][class_name] = threshold_list[threshold_idx]['F1_score']
            F1_score['dv_F1_score'][class_name] = threshold_list[threshold_idx]['dv_F1_score']
            
        return F1_score
            
    
    def compute_mAP(self):  
        AP = self.compute_PR_area()     
        mAP_dict = dict()
        for key, class_ap in AP.items():
            sum_AP = 0
            for class_name, AP in class_ap.items():    
                assert AP <=1.0
                sum_AP +=AP
            mAP = sum_AP/len(self.model.CLASSES)
            if key == 'classes_AP': mAP_dict['mAP'] = round(mAP, 4) 
            elif key == 'classes_dv_AP': mAP_dict['dv_mAP'] = round(mAP, 4) 
            
        return mAP_dict
      
    def compute_PR_area(self):
        AP = dict(classes_AP = dict(),
                  classes_dv_AP = dict())
        for class_name, threshold_list in self.precision_recall_dict.items():
            PR_list, dv_PR_list = [], []
            for idx, threshold in enumerate(threshold_list):
                PR_list.append([threshold['precision'], threshold['recall']])
                dv_PR_list.append([threshold['dv_precision'], threshold['dv_recall']])
            PR_list.sort(key = lambda x : x[1])     # Sort by recall
            dv_PR_list.reverse()  
                
            # adjust sum of recall
            before_recall = sum_recall = 0
            for idx, precision_recall in enumerate(dv_PR_list):
                _, recall = precision_recall
                sum_recall +=abs(recall - before_recall)
                before_recall = recall
                
            adjust_value = 1
            if sum_recall > 1:
                # ideally, the values of recall are sorted in ascending order
                # but there are cases where the value rises and then falls.
                #   (by compute iou by divied polygons)
                # In this case, `sum_recall` exceed 1 sometimes.
                # to adjust this `sum_recall`, we need `adjust_value`
                adjust_value  = 1/sum_recall  
            
            
            def compute_PR_area_(PR_list, adjust_value = 1):
                ap_area = 0
                before_recall =0
                continue_count = -1
                for idx, precision_recall in enumerate(PR_list):
                    if continue_count > 0: 
                        continue_count -=1 
                        if continue_count == 0 : continue_count = -1
                        continue
                    precision, recall = precision_recall
                    
                    # TODO: 계산법 다시
                    # idx+1 < len(PR_list) : if not last 
                    # PR_list[idx+1][1] == recall : if recall eqaul value as next recall
                    if idx+1 < len(PR_list) and PR_list[idx+1][1] == recall:
                        tmp_precision_list = []
                        # searching all next PR_list values 
                        # and only the precision when the recall is the same value is selected.
                        for i in range(idx, len(PR_list)):
                            if recall == PR_list[i][1]:       # same recall
                                continue_count +=1
                                tmp_precision_list.append(PR_list[i][0])
                            else: break
                        # get the largest value among selected precisions
                        precision = max([precis for precis in tmp_precision_list])
                    
                    # print(f"before_recall : {before_recall:.2f}, recall : {recall:.2f}    {abs(before_recall - recall):.2f}
                    # precision: {precision:.2f}     area = {abs(before_recall - recall)*precision:.2f}")
                    area = abs(recall - before_recall)*precision
                    if adjust_value != 1:
                        area = area*adjust_value
                    ap_area += area
                    before_recall = recall
                
                return ap_area
            
            dv_ap_area = compute_PR_area_(dv_PR_list, adjust_value)
            ap_area = compute_PR_area_(PR_list,)
            
            # print(f"{class_name}, dv_ap_area : {dv_ap_area}")
            # print(f"{class_name}, ap_area : {ap_area}")
            AP['classes_dv_AP'][class_name] = round(dv_ap_area, 4)
            AP['classes_AP'][class_name] = round(ap_area, 4)
        return AP
   
    
    def compute_precision_recall(self):
        self.precision_recall_dict = self.confusion_matrix.copy()       # for contain recall, precision, F1_score
    
        for class_name, threshold_list in self.confusion_matrix.items():
            for trsh_idx, threshold in enumerate(threshold_list):
                
                # compute recall
                if threshold['num_gt'] == 0:    # class: `class_nema` is not exist in dataset
                    recall = dv_recall = 0
                else:  
                    recall = threshold['num_true']/threshold['num_gt']
                    dv_recall = threshold['num_dv_true']/threshold['num_gt']
                 
                # compute precision
                if threshold['num_pred'] == 0: 
                    precision = dv_precision = 0
                else: 
                    if threshold['num_dv_pred'] == 0:   dv_precision = 0
                    else:   dv_precision = threshold['num_dv_true']/threshold['num_dv_pred']
                    precision = threshold['num_true']/threshold['num_pred']
                
                self.precision_recall_dict[class_name][trsh_idx]['recall'] = recall
                self.precision_recall_dict[class_name][trsh_idx]['precision'] = precision
                # compute F1_score with not divided polygons
                if recall == 0 and precision == 0: self.precision_recall_dict[class_name][trsh_idx]['F1_score'] =0
                else: self.precision_recall_dict[class_name][trsh_idx]['F1_score'] = 2*(precision*dv_recall)/(precision+dv_recall)
                
                
                self.precision_recall_dict[class_name][trsh_idx]['dv_recall'] = dv_recall
                self.precision_recall_dict[class_name][trsh_idx]['dv_precision'] = dv_precision
                # compute F1_score with divided polygons
                if dv_recall == 0 and dv_precision == 0: self.precision_recall_dict[class_name][trsh_idx]['dv_F1_score'] =0
                else: self.precision_recall_dict[class_name][trsh_idx]['dv_F1_score'] = 2*(dv_precision*dv_recall)/(dv_precision+dv_recall)
        

    def get_precision_recall_value(self):
        for class_name in self.classes:
            self.confusion_matrix[class_name] = []
            for i in range(len(self.iou_threshold)):
                self.confusion_matrix[class_name].append(dict(iou_threshold = self.iou_threshold[i],
                                                num_gt = 0,         # number of ground truth object
                                                num_dv_pred = 0,    # number of predicted objects by divided polygons
                                                num_dv_true = 0,    # successfully predicted objects among predicted objects by divided polygons,
                                                num_pred = 0,       # number of predicted objects 
                                                num_true = 0)       # successfully predicted objects among predicted objects
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
            
            
            self.get_num_pred_truth(gt_dict, infer_dict)  
    
    
    
    def get_num_pred_truth(self, gt_dict, infer_dict, window_num = 3):
        """
            count of 'predicted object' and 'truth predicted object'
        """
        num_predicted_object = len(infer_dict['bboxes'])                    
        num_gt = len(gt_dict['bboxes'])
        for idx, threshold in enumerate(self.iou_threshold):        
            done_gt = []        
            for i in range(num_predicted_object):
                pred_class_name = self.classes[infer_dict['labels'][i]]
                for j in range(num_gt):
                    gt_class_name = self.classes[gt_dict['labels'][j]]
                    
                    # number of gt of each threshold by class_name
                    if i == 0: self.confusion_matrix[gt_class_name][idx]['num_gt'] +=1 
                                    
                    if j in done_gt: continue
                    
                    # compute intersection over union
                    i_bboxes, gt_bboxes = infer_dict['bboxes'][i], gt_dict['bboxes'][j]
                    iou = compute_iou(i_bboxes, gt_bboxes)
                    
                    
                    if (iou > threshold and          
                        infer_dict['score'][i] > self.cfg.confidence_thrs):
                        self.confusion_matrix[pred_class_name][idx]['num_pred'] +=1
                        if pred_class_name == gt_class_name:  
                            self.confusion_matrix[pred_class_name][idx]['num_true'] +=1
                        
                        # compute iou by sliced polygon 
                        i_polygons, gt_polygons = infer_dict['polygons'][i], gt_dict['polygons'][j]
                        gt_xsort_bbox_list, gt_ysort_bbox_list = get_divided_polygon(gt_polygons, window_num)
                        i_xsort_bbox_list, i_ysort_bbox_list = get_divided_polygon(i_polygons, window_num)
                    
                        for i_xsort_bbox, gt_xsort_bbox in zip(i_xsort_bbox_list, gt_xsort_bbox_list):
                            if compute_iou(i_xsort_bbox, gt_xsort_bbox) < threshold:  continue
                        for i_ysort_bbox, gt_ysort_bbox in zip(i_ysort_bbox_list, gt_ysort_bbox_list):
                            if compute_iou(i_ysort_bbox, gt_ysort_bbox) < threshold:  continue
                        
                        # for calculate recall by divided_polygon
                        # number of predicted objects
                        self.confusion_matrix[pred_class_name][idx]['num_dv_pred'] +=1
                        
                        if pred_class_name == gt_class_name: 
                            # successfully predicted objects among predicted objects
                            self.confusion_matrix[pred_class_name][idx]['num_dv_true'] +=1
                        
                        done_gt.append(j)
                            
        
                        
 
