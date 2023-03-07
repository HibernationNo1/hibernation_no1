from hibernation_no1.mmdet.visualization import mask_to_polygon
from hibernation_no1.mmdet.inference import parse_inference_result

class Get_info():
    def __init__(self, 
                 bboxes, labels,
                 classes, score_thr = 0.5, kwargs = dict()):
        self.bboxes = bboxes
        self.labels = labels

        self.classes = classes
        self.score_thr = score_thr
        self.perse_bbox_label()
        self.kwargs = kwargs
            

    def perse_bbox_label(self):
        self.bboxes_list, self.labels_list = [], []

        bboxes = self.bboxes
        labels = self.labels  
 
        assert bboxes is None or bboxes.ndim == 2, \
            f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
        assert labels.ndim == 1, \
            f' labels ndim should be 1, but its ndim is {labels.ndim}.'
        assert bboxes is None or bboxes.shape[1] == 4 or bboxes.shape[1] == 5, \
            f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'
        assert bboxes is None or bboxes.shape[0] <= labels.shape[0], \
            'labels.shape[0] should not be less than bboxes.shape[0].'   

        if self.score_thr > 0:
            assert bboxes is not None and bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > self.score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]
                
        
        scores = bboxes[:, -1]      # [num_instance]
        bboxes = bboxes[:, :4]      # [num_instance, [x_min, y_min, x_max, y_max]]
        
        for bbox, label in zip (bboxes, labels):
            bbox_int = []
            for ele in bbox:
                bbox_int.append(int(ele))

            self.bboxes_list.append(bbox_int)
            self.labels_list.append(self.classes[label])

    def get_board_info(self):
        board_type = ['r_board', 'l_board']
        number_box = ['r_m_n', 'r_s_n', 'l_m_n', 'l_s_n']

        self.board_idx_list, self.number_box_idx_list, self.text_idx_list = [], [], []
        for i, b_l in enumerate(zip(self.bboxes_list, self.labels_list)):
            bbox, label = b_l

            if label in board_type:
                self.board_idx_list.append(i)
            elif label in number_box:
                self.number_box_idx_list.append(i)
            else:
                self.text_idx_list.append(i)
        
        return self.get_numberboard_info()


    def get_numberboard_info(self):        
        number_board_list = []
        for board_idx in self.board_idx_list:
            board_bbox = self.bboxes_list[board_idx]
            board_dict = dict(type = self.labels_list[board_idx],
                              board_center_p = self.compute_center_point(board_bbox),
                              width = self.compute_width_height(board_bbox)[0],
                              height = self.compute_width_height(board_bbox)[1])

            for number_box_idx in self.number_box_idx_list:
                number_box_bbox = self.bboxes_list[number_box_idx]
        
                if self.check_in_bbox(board_bbox, number_box_bbox):
                    
                    text_list = []
                    for text_idx in self.text_idx_list:
                        text_bbox = self.bboxes_list[text_idx]
                        # Check if text is located inside the number_box_bbox
                        if self.check_in_bbox(number_box_bbox, text_bbox):
                            x_center, _ = self.compute_center_point(text_bbox)
                            text_list.append([text_bbox, self.labels_list[text_idx], x_center])
                         
                    if self.labels_list[number_box_idx] in ['l_s_n', 'r_s_n']:                        
                        if len(text_list) !=3: continue  
                        if text_list[-1][1].isdigit(): continue
                        board_dict['sub_text'] = [i[1] for i in text_list]

                    elif self.labels_list[number_box_idx] in ['l_m_n', 'r_m_n']:
                        if len(text_list) !=4: continue 
                        isdigit = True
                        for text in text_list:
                            if not text[1].isdigit(): 
                                isdigit == False
                                break
                        if not isdigit: continue
                        board_dict['main_text'] = [i[1] for i in text_list]
               
                
            if board_dict.get('main_text', None) is not None and\
                board_dict.get('sub_text', None):
                number_board_list.append(board_dict)
        
        return number_board_list
               
            
    def check_in_bbox(self, out_box, in_box, box_inner_ratio = 0.0):
        x_min_out, y_min_out, x_max_out, y_max_out = out_box
        width_out, height_out = self.compute_width_height(out_box)


        x_center_in, y_center_in = self.compute_center_point(in_box)

        if x_min_out + (width_out * box_inner_ratio) <= x_center_in and\
            y_min_out + (height_out * box_inner_ratio)<= y_center_in and\
            x_max_out - (width_out * box_inner_ratio)>= x_center_in and\
            y_max_out - (height_out * box_inner_ratio)>= y_center_in:
            return True
        else: return False

    def compute_width_height(self, bbox):
        x_min, y_min, x_max, y_max = bbox
        return x_max - x_min, y_max-y_min        

    def compute_center_point(self, bbox):
        x_min, y_min, x_max, y_max = bbox

        x_center, y_center = int((x_min + x_max)/2), int((y_min + y_max)/2)

        return x_center, y_center