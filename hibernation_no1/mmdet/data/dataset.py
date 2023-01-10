from torch.utils.data import Dataset
import json
import os, os.path as osp
import numpy as np
from terminaltables import AsciiTable

from hibernation_no1.mmdet.data.transforms.compose import Compose
from hibernation_no1.mmdet.data.api.coco import COCO


def _build_dataset(dataset_cfg, dataset_api):
    if dataset_cfg is None: return None
    else: return CustomDataset(dataset_api = dataset_api, **dataset_cfg)
        
def build_dataset(train_cfg = None, val_cfg = None, dataset_api = 'coco'):
    train_dataset = _build_dataset(train_cfg, dataset_api)
    val_dataset = _build_dataset(val_cfg, dataset_api)    

    if train_dataset is not None and val_dataset is None:   return train_dataset, None        # only train dataset
    elif val_dataset is not None and train_dataset is None: return None, val_dataset          # only val dataset
    else: return train_dataset, val_dataset     
    
    

class CustomDataset(Dataset):
    """Custom dataset for detection.

    The annotation format is shown as follows. The `ann` field is optional for
    testing.

    .. code-block:: none

        [
            {
                'filename': 'a.jpg',
                'width': 1280,
                'height': 720,
                'ann': {
                    'bboxes': <np.ndarray> (n, 4) in (x1, y1, x2, y2) order.
                    'labels': <np.ndarray> (n, ),
                    'bboxes_ignore': <np.ndarray> (k, 4), (optional field)
                    'labels_ignore': <np.ndarray> (k, 4) (optional field)
                }
            },
            ...
        ]

    Args:
        ann_file (str): Annotation file path.
        pipeline (list[dict]): Processing pipeline.
        classes (str | Sequence[str], optional): Specify classes to load.
            If is None, ``cls.CLASSES`` will be used. Default: None.
        data_root (str, optional): Data root for ``ann_file``,
            ``img_prefix``, ``seg_prefix``, ``proposal_file`` if specified.
        test_mode (bool, optional): If set True, annotation will not be loaded.
        filter_empty_gt (bool, optional): If set true, images without bounding
            boxes of the dataset's classes will be filtered out. This option
            only works when `test_mode=False`, i.e., we never filter images
            during tests.
    """

    CLASSES = None    
        
    PALETTE = None

    def __init__(self,
                 dataset_api = "coco",
                 ann_file=None,
                 pipeline=None,
                 data_root=None,
                 img_prefix=None,
                 classes=None,
                 filter_empty_gt=True):
        
        self.dataset_api = dataset_api
        if self.confirm_return([ann_file, pipeline, data_root, img_prefix]):
            self.data_root = data_root if osp.isabs(data_root) else  osp.join(os.getcwd(), data_root) 
            self.ann_file = ann_file if osp.isabs(ann_file) else  osp.join(self.data_root, ann_file)
            self.img_prefix = img_prefix if osp.isabs(img_prefix) else  osp.join(self.data_root, img_prefix)        
            self.filter_empty_gt = filter_empty_gt
            assert osp.isfile(self.ann_file), f"The file: {self.ann_file} dose not exist."
            assert osp.isdir(self.data_root), f"The directory: {self.data_root} dose not exist."
            assert osp.isdir(self.img_prefix), f"The directory: {self.img_prefix} dose not exist."
        
            with open(self.ann_file, "r") as file:
                self.data_ann = json.load(file)
            
            self.CLASSES = self.get_classes(self.data_ann, classes)
            self.PALETTE = self.get_palette()
            self.data_infos = self.load_annotations(self.ann_file)        
                    
            self.pipeline = Compose(pipeline)
        
            
            valid_inds = self._filter_imgs()    # discard image without instances
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            
            # set group flag for the sampler
            self._set_group_flag()  
        else:
            pass
        
        
    def confirm_return(self, arg_list):
        for arg in arg_list:
            if arg is None: return False
        return True
    
        
    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)


    def get_palette(self):
        if self.CLASSES is not None:
            # just for visualization. no relationship with classification loss.
            palette = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
                    (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
                    (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
                    (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
                    (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
                    (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
                    (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
                    (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255),
                    (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255),
                    (134, 134, 103), (145, 148, 174), (255, 208, 186),
                    (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255),
                    (151, 0, 95), (9, 80, 61), (84, 105, 51), (74, 65, 105),
                    (166, 196, 102), (208, 195, 210), (255, 109, 65), (0, 143, 149),
                    (179, 0, 194), (209, 99, 106), (5, 121, 0), (227, 255, 205),
                    (147, 186, 208), (153, 69, 1), (3, 95, 161), (163, 255, 0),
                    (119, 0, 170), (0, 182, 199), (0, 165, 120), (183, 130, 88),
                    (95, 32, 0), (130, 114, 135), (110, 129, 133), (166, 74, 118),
                    (219, 142, 185), (79, 210, 114), (178, 90, 62), (65, 70, 15),
                    (127, 167, 115), (59, 105, 106), (142, 108, 45), (196, 172, 0),
                    (95, 54, 80), (128, 76, 255), (201, 57, 1), (246, 0, 122),
                    (191, 162, 208)]
            
            tmp_palette = []
            for i in range(len(self.CLASSES)):
                tmp_palette.append(palette[i])
            return tmp_palette
        return None
    

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        if self.dataset_api in ["coco", "COCO"]:
            self.coco = COCO(ann_file)
            # The order of returned `cat_ids` will not
            # change with the order of the CLASSES
        else: 
            raise ValueError(f"This dataset only support coco dataset.")
      

        # for using custom dataset
        with open(ann_file, "r") as file:
            data_ann = json.load(file)
            
        if data_ann['info']['description'] == 'Hibernation Custom Dataset':     #
            self.cat_ids = []                                                   #
            for cat_dict in data_ann['categories']:                             #
                self.cat_ids.append(cat_dict['id'])                             #
        else :                                                                  #
            self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
    
        data_infos = []
        total_ann_ids = []
        for d, i in enumerate(self.img_ids):
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)                     # [{'license', 'file_name', 'coco_url', 'height', 'width', 'date_captured', 'flickr_url', 'id', 'filename'}]
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)

        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
      
        return data_infos
    
    
    # print dataset to table
    # example
    # >>>  ustomDataset train dataset with number of images 1896, and instance counts:
    # >>>  +------------+-------+---------------+-------+-----------+-------+-----------+-------+-----------+-------+
    # >>>  | category   | count | category      | count | category  | count | category  | count | category  | count |
    # >>>  +------------+-------+---------------+-------+-----------+-------+-----------+-------+-----------+-------+
    # >>>  | 0 [obj_0]  | n_0   | 1 [obj_1]     | n_1   | 2 [obj_2] | n_2   | 3 [obj_3] | n_3   | 4 [obj_4] | n_4   |
    # >>>  |            |       |               |       |           |       |           |       |           |       |
    # >>>  | 5 [obj_5]  | n_5   |               |       |           |       |           |       |           |       |
    # >>>  +------------+-------+---------------+-------+-----------+-------+-----------+-------+-----------+-------+
    def __repr__(self):
        """Print the number of instance number."""
        # return dataset to (build_from_cfg) of (build_dataset) in dataset>builder.py
   
        result = (f'\n{self.__class__.__name__} train dataset '
                  f'with number of images {len(self)}, '
                  f'and instance counts: \n')
        
        if self.CLASSES is None:
            result += 'Category names are not provided. \n'
            return result
        
        instance_count = np.zeros(len(self.CLASSES) + 1).astype(int)
    
        # count the instance number in each image
        for idx in range(len(self)):
            label = self.get_ann_info(idx)['labels']  
            unique, counts = np.unique(label, return_counts=True)  
            if len(unique) > 0:
                # add the occurrence number to each class
                instance_count[unique] += counts
            else:
                if len(label) == 0: continue
                # background is the last index
                instance_count[-1] += 1   
                
        # create a table with category count
        table_data = [['category', 'count'] * 5]
        row_data = []

        for cls, count in enumerate(instance_count):
            if cls < len(self.CLASSES):
                row_data += [f'{cls} [{self.CLASSES[cls]}]', f'{count}']
            else:
                # add the background number
                row_data += ['-1 background', f'{count}']  
            if len(row_data) == 10:
                table_data.append(row_data)
                row_data = []
        if len(row_data) >= 2:
            if row_data[-1] == '0':
                row_data = row_data[:-2]
            if len(row_data) >= 2:
                table_data.append([])
                table_data.append(row_data)

    
        table = AsciiTable(table_data)
        result += table.table        
        return result
    

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)


    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann


    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        # obtain images that contain annotation
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.coco.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann
        
        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids[i]
            if self.filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds
    

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """ 
        self.flag = np.zeros(len(self), dtype=np.uint8)

        for i in range(len(self)):
            img_info = self.data_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    
    def get_classes(cls, data_ann, classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        Returns:
            tuple[str] or list[str]: Names of categories of the dataset.
        """
        if classes is None:   
            classes = data_ann['classes']
            return classes             
            

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """
        while True:
            data = self.prepare_train_img(idx)    
            if data is None:
                idx = self._rand_another(idx)
                continue
                
            return data
    
    def _rand_another(self, idx):
        """Get another random index from the same group as the given index."""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        
        self.pre_pipeline(results)
        return self.pipeline(results)
        
    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.img_prefix     # directory path where images are located
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        
        
    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by \
                pipeline.
        """

        img_info = self.data_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        
       
        return self.pipeline(results)