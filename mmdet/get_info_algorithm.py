from hibernation_no1.mmdet.visualization import mask_to_polygon
from hibernation_no1.mmdet.inference import parse_inferece_result

class Get_info():
    def __init__(self, results, classes, score_thr = 0):
        self.results = results
        self.classes = classes
        self.score_thr = score_thr
        self.get_mask_bbox_label()
            

    def get_mask_bbox_label(self):
        self.bboxes, self.labels, self.polygons = [], [], []

        for result in self.results:
            bboxes, labels, masks = parse_inferece_result(result)    

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
            
            if self.score_thr > 0:
                assert bboxes is not None and bboxes.shape[1] == 5
                scores = bboxes[:, -1]
                inds = scores > self.score_thr
                bboxes = bboxes[inds, :]
                labels = labels[inds]
                if masks is not None:
                    masks = masks[inds, ...]
                    
            
            scores = bboxes[:, -1]      # [num_instance]
            bboxes = bboxes[:, :4]      # [num_instance, [x_min, y_min, x_max, y_max]]
            polygons = mask_to_polygon(masks)

            for bbox, label, polygon in zip (bboxes, labels, polygons):
                bbox_int = []
                for ele in bbox:
                    bbox_int.append(int(ele))

                self.bboxes.append(bbox_int)
                self.polygons.append(polygon)
                self.labels.append(self.classes[label])


    def get_board_info(self):
        board_type = ['r_board', 'l_board']
        number_box = ['r_m_n', 'r_s_n', 'l_m_n', 'l_s_n']

        self.board_idx_list, self.number_box_idx_list, self.text_idx_list = [], [], []
        for i, b_l in enumerate(zip(self.bboxes, self.labels)):
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
            board_bbox = self.bboxes[board_idx]
            board_dict = dict()

            for number_box_idx in self.number_box_idx_list:
                number_box_bbox = self.bboxes[number_box_idx]

                if self.check_in_bbox(board_bbox, number_box_bbox):
                    board_dict['type'] = self.labels[board_idx]
                    
                    text_list = []
                    for text_idx in self.text_idx_list:
                        text_bbox = self.bboxes[text_idx]
                        if self.check_in_bbox(number_box_bbox, text_bbox):
                            x_center, _ = self.compute_center_point(text_bbox)
                            text_list.append([text_bbox, self.labels[text_idx], x_center])
                    
                    text_list.sort(key = lambda x: x[2])
                  
                    if self.labels[number_box_idx] in ['l_s_n', 'r_s_n']:
                        if len(text_list) !=3: continue     
                        if text_list[-1][1].isdigit(): continue
                        board_dict['sub_text'] = [i[1] for i in text_list]

                    elif self.labels[number_box_idx] in ['l_m_n', 'r_m_n']:
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
               
            
    def check_in_bbox(self, out_box, in_box):
        x_min_out, y_min_out, x_max_out, y_max_out = out_box
        x_center_in, y_center_in = self.compute_center_point(in_box)

        if x_min_out <= x_center_in and\
            y_min_out <= y_center_in and\
            x_max_out >= x_center_in and\
            y_max_out >= y_center_in:
            return True
        else: return False


    def compute_center_point(self, bbox):
        x_min, y_min, x_max, y_max = bbox

        x_center, y_center = int((x_min + x_max)/2), int((y_min + y_max)/2)

        return x_center, y_center