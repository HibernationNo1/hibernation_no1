import numpy as np

from torch.utils.data import Sampler



class GroupSampler(Sampler):
    def __init__(self, dataset, batch_size=1):
        assert hasattr(dataset, 'flag')  
        self.batch_size = batch_size
        
        # flag for the higher value between image's width and height  
        self.flag = dataset.flag.astype(np.int64)       # [0 or 1, 0 or 1, ... 0 or 1]  0 : width > height, 1 : width < height
        self.group_sizes = np.bincount(self.flag)       # [count of 0, count of 1]
        self.num_samples = 0
        for size in self.group_sizes:
            self.num_samples += int(np.ceil(size / self.batch_size)) * self.batch_size
    
    def __iter__(self):
        indices = []        # will be contains each image index 
        
        for i, size in enumerate(self.group_sizes):
            # image를 width, height 기준으로 나뉘어진 group
            if size == 0:
                continue
            indice = np.where(self.flag == i)[0]        
            assert len(indice) == size
            np.random.shuffle(indice)                           # apply shuffle 
            num_extra = int(np.ceil(size / self.batch_size)     # remaining number for batch size
                            ) * self.batch_size - len(indice)
            indice = np.concatenate(                                # seletec image rendomly as `num_extra`
                [indice, np.random.choice(indice, num_extra)])
            indices.append(indice)                                  # append for get lenght equal to batch size
        
        indices = np.concatenate(indices)
        indices = [     # list of list with length batch_size
            indices[i * self.batch_size:(i + 1) * self.batch_size]
            for i in np.random.permutation(
                range(len(indices) // self.batch_size))
        ]
        # len(indices) == round(number of image / batch_size, 1)
        indices = np.concatenate(indices)
        indices = indices.astype(np.int64).tolist()         # set list to 1 dimention 

        assert len(indices) == self.num_samples
        return iter(indices)
            
    
    def __len__(self):
        return self.num_samples