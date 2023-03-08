import numpy as np
import os, os.path as osp
import cv2
import torch
import warnings
from hibernation_no1.mmdet.inference import inference_detector, parse_inference_result
from hibernation_no1.mmdet.visualization import mask_to_polygon, draw_PR_curve, draw_to_img
from hibernation_no1.mmdet.get_info_algorithm import Get_info

def compute_iou(infer_box, gt_box, confidence_score, img = None):
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
    iou = inter / outer * confidence_score
    
    # if img is not None:
    #     if iou > 0.5:
    #         print(f"confidence_score : {confidence_score}")
    #         print(f"outer : {outer}, inter ; {inter}")
    #         print(f"inter / outer * confidence_score : {inter / outer * confidence_score} ")
    #         print(f"inter / outer  : {inter / outer}")
    #         cv2.rectangle(img, pt1 = (int(x1), int(y1)), pt2 = (int(x2), int(y2)), color = (255, 255, 0), thickness = 2)      
    #         infer_left_top =  [int(infer_box[0]), int(infer_box[1])]    
    #         infer_right_bottom =  [int(infer_box[2]), int(infer_box[3])] 
    #         cv2.rectangle(img, pt1 = infer_left_top, pt2 = infer_right_bottom, color = (255, 0, 0), thickness = 1)          
    #         gt_left_top =  [int(gt_box[0]), int(gt_box[1])]    
    #         gt_right_bottom =  [int(gt_box[2]), int(gt_box[3])] 
    #         cv2.rectangle(img, pt1 = gt_left_top, pt2 = gt_right_bottom, color = (0, 0, 255), thickness = 1)  

    #         print(f"iou ; {iou}")
    #         cv2.imshow("img", img)
    #         while True:
    #             if cv2.waitKey() == 27: break


    return iou


def get_divided_polygon(polygon, window_num, min_num_points = 10):
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

    # return None if number of points of polygon less than `min_num_points` 
    if len(polygon) < min_num_points: return None    
    # return None if (number of points of polygon//window_num) less than `min_num_points`   
    if len(polygon) // window_num < min_num_points : return None
    
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
        

def get_distance(point_1, point_2):
    return math.sqrt(math.pow(point_1[0] - point_2[0], 2) + math.pow(point_1[1] - point_2[1], 2))
                

class Evaluate():
    def __init__(self, model, cfg, dataloader, output_path = None, **kwargs):
        self.model = model
        self.cfg = cfg
        self.dataloader = dataloader
        self.classes = self.model.CLASSES
        self.confusion_matrix = dict()
        self.kwargs = kwargs
        self.output_path = output_path
        self.plot_dir = "plots"
        self.img_result_dir = "images"

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
        assert len(self.iou_threshold) > threshold_idx
        
        F1_score = dict(F1_score = dict(),
                        dv_F1_score = dict())
        for class_name, threshold_list in self.precision_recall_dict.items():
            F1_score['F1_score'][class_name] = threshold_list[threshold_idx]['F1_score']
            F1_score['dv_F1_score'][class_name] = threshold_list[threshold_idx]['dv_F1_score']
            
        return F1_score
    

    def save_PR_curve(self, out_path):
        for class_name, PR_curve_values in self.PR_curve_values.items():
            RP_out_path = osp.join(out_path, f'RP_curve_{class_name}.jpg')
            draw_PR_curve(RP_out_path,
                          class_name, 
                          PR_curve_values['PR_list'],
                          self.PR_curve_values[class_name]['ap_area'],
                          show_plot = self.cfg.get('show_plot', False))

            dv_RP_out_path = osp.join(out_path, f'dv_RP_curve_{class_name}.jpg')
            draw_PR_curve(dv_RP_out_path,
                          class_name, 
                          PR_curve_values['dv_PR_list'],
                          self.PR_curve_values[class_name]['dv_ap_area'],
                          dv_flag = True,
                          show_plot = self.cfg.get('show_plot', False))

            


    def compute_mAP(self):  
        AP = self.compute_PR_area()     
        summary_dict = dict(normal = dict(),
                            dv = dict())
        for key, class_ap in AP.items():
            sum_AP = 0
            for class_name, AP in class_ap.items():    
                if AP >1.0: 
                    raise ValueError(f"key: {key}, class_name: {class_name}, AP: {AP}") 
                
                if key == 'classes_AP': summary_dict['normal'][f'{class_name} AP'] = round(AP, 4) 
                if key == 'classes_dv_AP': summary_dict['dv'][f'{class_name} AP'] = round(AP, 4) 
                sum_AP +=AP
            mAP = sum_AP/len(self.model.CLASSES)
            if key == 'classes_AP': summary_dict['normal']['mAP'] = round(mAP, 4) 
            elif key == 'classes_dv_AP': summary_dict['dv']['mAP'] = round(mAP, 4) 

        if (self.cfg.get('save_plot', False)) and (self.output_path is not None):
            if not osp.isdir(self.output_path):
                raise OSerror(f"The path is not exist!  \n path: {self.output_path}")

            plot_dir = osp.join(self.output_path, self.plot_dir)
            os.makedirs(plot_dir, exist_ok = True)
            self.save_PR_curve(plot_dir)

            
        return summary_dict
      

    def compute_PR_area(self):
        AP = dict(classes_AP = dict(),
                  classes_dv_AP = dict())

        for class_name, PR_curve_values in self.PR_curve_values.items():
            PR_list = PR_curve_values['PR_list']
            dv_PR_list = PR_curve_values['dv_PR_list']
 
            # adjust sum of recall
            before_recall_dv = sum_recall = 0
            for idx, precision_recall in enumerate(dv_PR_list):
                _, recall, _ = precision_recall
                sum_recall +=abs(recall - before_recall_dv)
                before_recall_dv = recall
                
            adjust_value = 1
            if sum_recall > 1:
                # ideally, the values of recall are sorted in ascending order
                # but there are cases where the value rises and then falls.
                #   (by compute iou by divied polygons)'recall
                # In this case, `sum_recall` exceed 1 sometimes.
                # to adjust this `sum_recall`, we need `adjust_value`
                adjust_value  = 1/sum_recall  
            
            
            def compute_PR_area_(input_PR_list, adjust_value = 1):
                max_pre, max_recall = -1, -1
                for PR in input_PR_list:
                    precision, recall, _ = PR
                    if max_pre < precision:
                        max_pre = precision
                    if max_recall < recall:
                        max_recall = recall
                if max_pre > 1 or max_recall >1 : 
                    raise ValueError(f"max value of precision or recall must be less than 1.0,"
                                     f" but got (precision, recall): ({max_pre, max_recall})")

                ap_area = 0
                before_recall, before_precision = 0, 0
                before_recall_dif = 0
                
                # print(F"\nclass_name: {class_name}")
                for idx, precision_recall in enumerate(input_PR_list):
                    precision, recall, threshold = precision_recall
                    # print(f"precision, recall, threshold : {precision:.4f}, {recall:.4f}, {threshold}")
                    if idx == 0:
                        area = (precision*recall)
                        if adjust_value != 1:
                            area = area*adjust_value
                        ap_area +=area
                        before_recall, before_precision = recall, precision
                        before_recall_dif = recall
                        # print(f"first area: {area:.4f},     ap_area ; {ap_area:.4f}")
                        continue
                    
                    # precision-recall curve is drawn from the right (high recall side)
                    # Because of this, before_recall have more large value
                    if before_recall > recall: 
                        raise ValueError("recall value should increases, but decreases\n"
                                         f"before_recall : {before_recall:.4f}, recall : {recall:.4f}")
                     
                   
                    # if recall remains the same and only precision increases
                    if before_recall == recall :
                        # print(f"before_recall == recall", end='')
                        if before_precision < precision:
                            # subtract previous value and add the current value
                            area = ((precision*before_recall_dif) - (before_precision*before_recall_dif))
                            if adjust_value != 1:
                                area = area*adjust_value
                            ap_area +=area
                            before_recall, before_precision = recall, precision
                            # print(f"    before_precision > precision, area : {area:.4f}     ap_area : {ap_area:.4f}")
                            continue
                        else:
                            # print(f"     esle area : {0}")
                            before_recall, before_precision = recall, precision
                            continue
                    
                    # precision-recall curve is drawn from the right (high recall side. from 1.0 to 0.0)    
                    before_recall_dif = recall_dif = recall - before_recall        
                    if before_precision < precision:        # increases precision
                        precision_dif = precision - before_precision
                        area = (recall_dif * precision) - (precision_dif*recall_dif/2)
                        # print(f"before_precision < precision, area : {area:.4f}", end = '')
                    elif before_precision == precision:     # precision remains the same
                        area = (precision *recall_dif)
                        # print(f"before_precision == precision, area : {area:.4f}", end = '')
                    elif before_precision > precision:      # decreases precision
                        precision_dif = before_precision - precision
                        area = (recall_dif * precision) + (precision_dif*recall_dif/2)
                        # print(f"before_precision > precision, area : {area:.4f}", end = '')
                     
                    before_recall, before_precision = recall, precision
                    if adjust_value != 1:
                        area = area*adjust_value
                    ap_area += area
                    # print(f"  ap_area : {ap_area:.4f}")
                
                if ap_area > 1.0:
                    raise ValueError(f"average precision must low than 1.0")
                return ap_area

            ap_area = round(compute_PR_area_(PR_list), 4)
            dv_ap_area = round(compute_PR_area_(dv_PR_list, adjust_value), 4)
            
            self.PR_curve_values[class_name]['ap_area'] = ap_area
            AP['classes_AP'][class_name] = ap_area
            
            self.PR_curve_values[class_name]['dv_ap_area'] = dv_ap_area
            AP['classes_dv_AP'][class_name] = round(dv_ap_area, 4)
            

           
        return AP
   
    
    def compute_precision_recall(self):
        self.precision_recall_dict = self.confusion_matrix.copy()       # for contain recall, precision, F1_score

        for class_name, threshold_list in self.confusion_matrix.items():
            for trsh_idx, threshold in enumerate(threshold_list):
                self.precision_recall_dict[class_name][trsh_idx]['threshold'] = threshold['iou_threshold']
                # compute recall
                if threshold['num_gt'] == 0:    # class: `class_name` is not exist in dataset
                    recall = dv_recall = 0
                else:  
                    recall = threshold['num_true']/threshold['num_gt']
                    dv_recall = threshold['num_dv_true']/threshold['num_gt']
                 
                # compute precision
                if threshold['num_pred'] == 0: 
                    precision = dv_precision = 1        # TODO: 1? 0?
                else: 
                    if threshold['num_dv_pred'] == 0:   dv_precision = 0
                    else:   dv_precision = threshold['num_dv_true']/threshold['num_dv_pred']
                    precision = threshold['num_true']/threshold['num_pred']                

                if recall > 1.0:  recall = 1.0
                if dv_recall > 1.0 : dv_recall = 1.0        # TODO


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
        
        self.PR_curve_values = dict()
        for class_name, info_dict_list in self.precision_recall_dict.items():
            PR_list, dv_PR_list = [], []
            for idx, info_dict in enumerate(info_dict_list):
                PR_list.append([info_dict['precision'], info_dict['recall'], info_dict['threshold']])
                dv_PR_list.append([info_dict['dv_precision'], info_dict['dv_recall'], info_dict['threshold']])

            PR_list.sort(key = lambda x : x[2], reverse= True)     # Sort by threshold
            dv_PR_list.sort(key = lambda x : x[2], reverse=True) 

            self.PR_curve_values[class_name] = dict(PR_list = PR_list,
                                                    dv_PR_list = dv_PR_list)


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
            # len(batch_gt_bboxes): batch_size 
            batch_gt_bboxes = val_data_batch['gt_bboxes'].data[0]
            batch_gt_labels = val_data_batch['gt_labels'].data[0]
            batch_gt_masks = val_data_batch['gt_masks'].data[0]
            batch_gts = []
            for gt_bboxes, gt_labels, gt_masks in zip(batch_gt_bboxes, batch_gt_labels, batch_gt_masks):
                batch_gts.append([gt_bboxes, gt_labels, gt_masks])
            
            batch_filepath = []
            for img_meta in val_data_batch['img_metas'].data[0]:
                batch_filepath.append(img_meta['file_path'])

            with torch.no_grad():
            # len: batch_size
                inference_detector_cfg = dict(model = self.model, 
                                              imgs_path = batch_filepath, 
                                              get_memory_info = self.kwargs.get('get_memory_info', None))
                batch_results = inference_detector(**inference_detector_cfg)  

            for results, ground_truths, file_path in zip(batch_results, batch_gts, batch_filepath):
                self.get_confusion_value(ground_truths, results, file_path)
                                        
        self.compute_precision_recall()
    



    def get_confusion_value(self, ground_truths, results, file_path):
        infer_bboxes, infer_labels, infer_masks = parse_inference_result(results) 
        gt_bboxes, gt_labels, gt_masks = ground_truths  

        
        if infer_masks is not None:
            show_score_thr = self.cfg.get('show_score_thr', 0)
           
            assert infer_bboxes is not None and infer_bboxes.shape[1] == 5
            scores = infer_bboxes[:, -1]
            if show_score_thr > 0:
                inds = scores > show_score_thr
            else:
                inds = scores > 0.5
            infer_bboxes = infer_bboxes[inds, :]
            infer_labels = infer_labels[inds]
            if infer_masks is not None:
                infer_masks = infer_masks[inds, ...]
            
            
            infer_scores = infer_bboxes[:, -1]      # [num_instance]
            infer_bboxes = infer_bboxes[:, :4]      # [num_instance, [x_min, y_min, x_max, y_max]]

            
            infer_polygons = mask_to_polygon(infer_masks)
            gt_polygons = mask_to_polygon(gt_masks.masks)
        else:   # detected nothing
            infer_scores = infer_bboxes = infer_polygons = gt_polygons = []

        
        infer_dict = dict(bboxes = infer_bboxes,
                          polygons = infer_polygons,
                          labels = infer_labels,
                          score = infer_scores)
        gt_dict = dict(bboxes = gt_bboxes,
                       polygons = gt_polygons,
                       labels = gt_labels)
            
        img = cv2.imread(file_path)

        self.get_num_pred_truth(gt_dict, infer_dict, num_window = self.cfg.num_window, img = img)  
    
    
    def get_num_pred_truth(self, gt_dict, infer_dict, num_window = 3, img = None):
        """
            count of 'predicted object' and 'truth predicted object'
        """

        inf_bboxes = infer_dict['bboxes']                 
        gt_bboxes = gt_dict['bboxes']

        for gt_i, gt_bbox in enumerate(gt_bboxes):
            gt_object_name = self.classes[gt_dict['labels'][gt_i]]
            for inf_i, inf_bbox in enumerate(inf_bboxes):
                inf_object_name = self.classes[infer_dict['labels'][inf_i]]
                confidence_score = infer_dict['score'][inf_i]

                iou = compute_iou(gt_bbox, inf_bbox, confidence_score, img = img) 

                for iou_i, iou_threshold in enumerate(self.iou_threshold):
                    
                    # count number of ground truth object of each iou threshold
                    if inf_i == 0:      
                        self.confusion_matrix[gt_object_name][iou_i]['num_gt'] +=1 

                    # should be upper than iou threshold
                    if iou < iou_threshold: continue  

                    # count number of predicted object of each iou threshold
                    # regardless of object name.
                    self.confusion_matrix[inf_object_name][iou_i]['num_pred'] +=1

                    # count number of predicted object of each iou threshold
                    # regard of object name.
                    if inf_object_name == gt_object_name:
                        # successfully predicted objects among predicted objects
                        self.confusion_matrix[inf_object_name][iou_i]['num_true'] +=1

                    # compute iou by sliced polygon 
                    # polygons : [[x_1, y_1], [x_2, y_2], ..., [x_n, y_n]]
                    inf_polygons, gt_polygons = infer_dict['polygons'][inf_i], gt_dict['polygons'][gt_i]

                   
                    gt_dv_bbox_list = get_divided_polygon(gt_polygons, num_window)
                    inf_dv_bbox_list = get_divided_polygon(inf_polygons, num_window)
                    if gt_dv_bbox_list is None or inf_dv_bbox_list is None:
                        continue    # cannot be divided polygon cause the number of points is too small.          
                    
                    # compute iou with each bbox of sliced of polygon 
                    dv_iou_flag = False
                    gt_xsort_bbox_list, gt_ysort_bbox_list = gt_dv_bbox_list
                    inf_xsort_bbox_list, inf_ysort_bbox_list = inf_dv_bbox_list
                    for inf_xsort_bbox, gt_xsort_bbox in zip(inf_xsort_bbox_list, gt_xsort_bbox_list):
                        if compute_iou(inf_xsort_bbox, gt_xsort_bbox, confidence_score) < iou_threshold:  
                            dv_iou_flag = True
                            break
                    for inf_ysort_bbox, gt_ysort_bbox in zip(inf_ysort_bbox_list, gt_ysort_bbox_list):
                        if compute_iou(inf_ysort_bbox, gt_ysort_bbox, confidence_score) < iou_threshold:  
                            dv_iou_flag = True
                            break
                    if dv_iou_flag: continue

                    # count number of predicted object of each iou threshold
                    # regardless of object name.
                    # computed with sliced of polygon
                    self.confusion_matrix[inf_object_name][iou_i]['num_dv_pred'] +=1
                    
                    # count number of predicted object of each iou threshold
                    # regard of object name.
                    # computed with sliced of polygon
                    if inf_object_name == gt_object_name:
                        # successfully predicted objects among predicted objects
                        self.confusion_matrix[inf_object_name][iou_i]['num_dv_true'] +=1


    def run_inference(self):
        # If self.output_path is None then the directory does not yet exist.
        if self.output_path is not None:
            img_result_dir = osp.join(self.output_path, self.img_result_dir)
            os.makedirs(img_result_dir, exist_ok = True)
        else: return None

        for i, val_data_batch in enumerate(self.dataloader):
            # len(batch_gt_bboxes): batch_size 
            batch_gt_bboxes = val_data_batch['gt_bboxes'].data[0]
            batch_gt_labels = val_data_batch['gt_labels'].data[0]
            batch_gts = []
            for gt_bboxes, gt_labels in zip(batch_gt_bboxes, batch_gt_labels):
                # append score
                gt_bboxes_scores = []
                for gt_bboxe in gt_bboxes.tolist():
                    gt_bboxe.append(100.)
                    gt_bboxes_scores.append(gt_bboxe)
                
                batch_gts.append([np.array(gt_bboxes_scores), gt_labels.numpy()])
            
            batch_filepath = []
            for img_meta in val_data_batch['img_metas'].data[0]:
                batch_filepath.append(img_meta['file_path'])

            with torch.no_grad():
            # len: batch_size
                inference_detector_cfg = dict(model = self.model, 
                                              imgs_path = batch_filepath, 
                                              get_memory_info = self.kwargs.get('get_memory_info', None))
                batch_results = inference_detector(**inference_detector_cfg)  

            for filepath, results, ground_truths in zip(batch_filepath, batch_results, batch_gts):
                bboxes, labels, masks = parse_inference_result(results) 
                print(f"mask : {masks}")
                print(f"type(masks) ; {type(masks)}")
                # Save the image with the inference result drawn
                img = cv2.imread(filepath)
                draw_cfg = dict(img = img,
                                bboxes = bboxes,
                                labels = labels,
                                masks = masks,
                                class_names = self.classes,
                                score_thr = self.cfg.get('show_score_thr', 0.5))
                img = draw_to_img(**draw_cfg)       # Draw bbox, seg, label and save drawn_img

                out_file = osp.join(img_result_dir, osp.basename(filepath))
                cv2.imwrite(out_file, img) 

                # Compute the ratio of how accurately the board's information was inferred 
                # by comparing the ground truth and the inference results.
                if self.cfg.get('compare_board_info', False): 
                    bboxes_gt, labels_gt = ground_truths 
                    correct_inference_rate = self.compare_board_info(bboxes, labels, bboxes_gt, labels_gt)
                    return correct_inference_rate
                else: 
                    return None


    
    def compare_board_info(self, bboxes_infer, labels_infer, bboxes_gt, labels_gt, distance_thr_rate = 0.1):    
        get_info_infer = Get_info(bboxes_infer, labels_infer,
                                  self.classes,
                                  score_thr = self.cfg.get('show_score_thr', 0.5))
        license_board_infer_list = get_info_infer.get_board_info()
        
        get_info_gt = Get_info(bboxes_gt, labels_gt,
                               self.classes, 
                               score_thr = self.cfg.get('show_score_thr', 0.5))
        license_board_gt_list = get_info_gt.get_board_info()
     
        matchs_count = 0
        for info_gt in license_board_gt_list:
            board_center_p_gt = info_gt.pop('board_center_p')
            board_width_p_gt = info_gt.pop('width')
            board_height_p_gt = info_gt.pop('height')
            for info_infer in license_board_infer_list:
                board_center_p_infer = info_infer.pop('board_center_p')
                board_width_p_infer = info_infer.pop('width')
                board_height_p_infer = info_infer.pop('height')

                if info_gt == info_infer :
                    # An inference can be considered correct 
                    # when the distance between the center points of the two boards is sufficiently close.
                    length_btw_board = get_distance(info_gt['board_center_p'], info_infer['board_center_p']) 
                    if length_btw_board < board_width_p_gt * distance_thr_rate and \
                       length_btw_board < board_height_p_gt * distance_thr_rate*2:
                       matchs_count+=1
        
        return matchs_count/len(license_board_gt_list)

                               
                                

                    

                