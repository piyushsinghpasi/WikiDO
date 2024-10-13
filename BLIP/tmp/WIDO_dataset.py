import os
import json

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from PIL import Image
import pandas as pd

from data.utils import pre_caption
import torch

# from utils import pre_caption

class WIDO_train(Dataset):
    def __init__(self, train_path, transform, image_root, max_words=30, prompt=''):        
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''        
        
        self.annotation = json.load(open(train_path,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt
        
        self.img_ids = {}  
        n = 0
        for ann in self.annotation:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):
        
        ann = self.annotation[index]
        #image_path = os.path.join(self.image_root,ann['image'])  
        image_path = self.image_root + ann['image']      
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        
        caption = self.prompt+pre_caption(ann['caption'], self.max_words) 

        return image, caption, self.img_ids[ann['image_id']] 
    
class WIDO_cap_eval(Dataset):
    def __init__(self, val_path, test_path, transform, image_root, split):  
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''

        filenames = {'val':val_path,'test':test_path}

        self.annotation = json.load(open(filenames[split],'r'))
        self.transform = transform
        self.image_root = image_root
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        #image_path = os.path.join(self.image_root,ann['image'])   
        image_path = self.image_root + ann['image']        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          
        
        #img_id = ann['image'].split('/')[-1].strip('.jpg').split('_')[-1]
        img_id = ann['image_id']
        
        return image, img_id
        
class coco_karpathy_retrieval_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split, max_words=30):  
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
       # urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json',
              #  'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json'}
        filenames = {'val':'coco_karpathy_val_small.json','test':'coco_karpathy_test_small.json'}
        
      #  download_url(urls[split],ann_root)
        
        self.annotation = json.load(open(os.path.join(ann_root,filenames[split]),'r'))
        self.transform = transform
        self.image_root = image_root
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption,max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
                                    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        image_path = os.path.join(self.image_root, self.annotation[index]['image'])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        return image, index
    
class WIDO_retrieval_eval(Dataset):
    def __init__(self, val_path, test_path, transform, image_root, split, max_words=30):  
        '''
        image_root (string): Root directory of images (e.g. flickr30k/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''

        filenames = {'val':val_path,'test':test_path}

        self.annotation = json.load(open(filenames[split],'r'))
        self.transform = transform
        self.image_root = image_root
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            self.text.append(pre_caption(ann['caption'],max_words))
            self.img2txt[img_id].append(txt_id)
            self.txt2img[txt_id] = img_id
            txt_id += 1
                                    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        image_path = self.image_root +  self.annotation[index]['image']        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        return image, index, image_path   
    
    
class WIDO_train_codebook(Dataset):
    def __init__(self, train_path, transform, image_root, max_words=30, prompt=''):        
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''        
        
        self.annotation = json.load(open(train_path,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt
        
        self.img_ids = {}  
        n = 0
        for ann in self.annotation:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):
        
        ann = self.annotation[index]
        #image_path = os.path.join(self.image_root,ann['image'])  
        image_path = self.image_root + ann['image']      
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        
        caption = self.prompt+pre_caption(ann['caption'], self.max_words) 
        topic = ann['topic']

        return image, caption, self.img_ids[ann['image_id']], topic  
    
class WIDO_eval_codebook(Dataset):
    def __init__(self, val_path, test_path, transform, image_root, split, max_words=30):  
        '''
        image_root (string): Root directory of images (e.g. flickr30k/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''

        filenames = {'val':val_path,'test':test_path}

        self.annotation = json.load(open(filenames[split],'r'))
        self.transform = transform
        self.image_root = image_root
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        self.topic = []
        
        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            self.text.append(pre_caption(ann['caption'],max_words))
            self.topic.append(ann['topic'])
            self.img2txt[img_id].append(txt_id)
            self.txt2img[txt_id] = img_id
            txt_id += 1
                                    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        image_path = self.image_root +  self.annotation[index]['image']        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  
        topic = self.annotation[index]['topic']
        max_words = 30
        caption = pre_caption(self.annotation[index]['caption'],max_words)

        return image, index, image_path, caption, topic 
    
     

class NegativeSampler():
    def __init__(self, df: pd.DataFrame, num_negatives: int, sampling_type: str, seed: int = 1):
        """Sample Negatives

        Args:
            df (pd.DataFrame): dataframe containing image_path, caption and topic/class
            num_negatives (int): no. of negatives to sample
            sampling_type (str): type of sampling. Options 
            seed (int, optional): for reproducibility during sampling. Defaults to 1.
        """
        self.df = df # assumes contains topic/ self.class_column
        self.num_negatives = num_negatives
        self.class_column = "topic" # df column that contains class 
        self.seed = seed 

        # all classes are equilikely and all instance per class are equilikely
        self.df['freq'] = 1./self.df.groupby(self.class_column)[self.class_column].transform('count')

        assert sampling_type in ["in_class", "diff_class"], f"sampling_type: {sampling_type} not in supported types [\"in_class\", \"diff_class\"]"
        self.sample = self.in_class_sampling if sampling_type == "in_class" else self.diff_class_sampling

    def in_class_sampling(self, index: int) -> pd.DataFrame:
        """return negatives samples from same class

        Args:
            index (int): index of curr instance in main Dataset class (take same one from __getitem__())

        Returns:
            pd.DataFrame: returns dataframe that contains negatives instances
        """

        # remove rest of the classes
        df_same_class = self.df[self.df[self.class_column] == self.df.iloc[index][self.class_column]]

        # negatives are sampled from other than curr index
        df_negatives = df_same_class[~df_same_class.index.isin([index])]

        negatives = df_negatives.sample(n = self.num_negatives, random_state = self.seed)

        return negatives
    
    def diff_class_sampling(self, index: int) -> pd.DataFrame:
        
        """return negatives samples from different class

        Args:
            index (int): index of curr instance in main Dataset class (take same one from __getitem__())

        Returns:
            pd.DataFrame: returns dataframe that contains negatives instances
        """
        # remove same class
        df_diff_class = self.df[self.df[self.class_column] != self.df.iloc[index][self.class_column]]

        # negatives are sampled from other than curr index
        df_negatives = df_diff_class[~df_diff_class.index.isin([index])]
        
        negatives = df_negatives.sample(n = self.num_negatives, random_state = self.seed, weights = self.df['freq'])

        return negatives
    
class WIDO_train_with_negatives(Dataset):
    def __init__(self, train_path, transform, image_root, max_words=30, prompt='',num_negatives=10, sampling_type="diff_class"):        
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''        
        
        self.annotation = json.load(open(train_path,'r'))

        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt
        
        self.img_ids = {}  
        n = 0
        for ann in self.annotation:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    

        df = pd.DataFrame(self.annotation)
        self.sampler = NegativeSampler(df, num_negatives = num_negatives, sampling_type = sampling_type)

    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):
        
        ann = self.annotation[index]
        #image_path = os.path.join(self.image_root,ann['image'])  
        pos_image = self.image_transform(ann['image'])
        pos_caption = self.caption_transform(ann['caption'])

        negatives = self.sampler.sample(index)

        neg_images = []
        neg_captions = []

        for idx, row in negatives.iterrows():
            neg_images.append(self.image_transform(row['image']).unsqueeze(0))
            neg_captions.append(self.caption_transform(row['caption']))
        
        neg_images = torch.cat(neg_images, dim=0)

        return pos_image, pos_caption, self.img_ids[ann['image_id']], neg_images, neg_captions
 
    def image_transform(self, image_path):
        image_path = self.image_root + image_path   
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        return image
    
    def caption_transform(self, caption):
        caption = self.prompt+pre_caption(caption, self.max_words)
        return caption
        
if __name__ == "__main__":
    data = [
        {"img_id":0, "image":"/topical_data/output_images/output_images/sublist_6/00002/000024248.jpg", "caption":"hallo", "topic":"a"},
        {"img_id":1, "image":"/topical_data/output_images/output_images/sublist_6/00002/000024248.jpg", "caption":"hallo", "topic":"a"},
        {"img_id":2, "image":"/topical_data/output_images/output_images/sublist_6/00002/000024248.jpg", "caption":"hallo", "topic":"a"},
        {"img_id":3, "image":"/topical_data/output_images/output_images/sublist_6/00002/000024248.jpg", "caption":"hallo", "topic":"a"},
        {"img_id":4, "image":"/topical_data/output_images/output_images/sublist_6/00002/000024248.jpg", "caption":"hallo", "topic":"a"},
        
        {"img_id":5, "image":"/topical_data/output_images/output_images/sublist_6/00002/000024248.jpg", "caption":"hallo", "topic":"b"},
        {"img_id":6, "image":"/topical_data/output_images/output_images/sublist_6/00002/000024248.jpg", "caption":"hallo", "topic":"b"},
        {"img_id":7, "image":"/topical_data/output_images/output_images/sublist_6/00002/000024248.jpg", "caption":"hallo", "topic":"b"},
        {"img_id":8, "image":"/topical_data/output_images/output_images/sublist_6/00002/000024248.jpg", "caption":"hallo", "topic":"b"},

        {"img_id":9, "image":"/topical_data/output_images/output_images/sublist_6/00002/000024248.jpg", "caption":"hallo", "topic":"c"},
        {"img_id":10, "image":"/topical_data/output_images/output_images/sublist_6/00002/000024248.jpg", "caption":"hallo", "topic":"c"},
        {"img_id":11, "image":"/topical_data/output_images/output_images/sublist_6/00002/000024248.jpg", "caption":"hallo", "topic":"c"},
        
        {"img_id":12, "image":"/topical_data/output_images/output_images/sublist_6/00002/000024248.jpg", "caption":"hallo", "topic":"d"},
        {"img_id":13, "image":"/topical_data/output_images/output_images/sublist_6/00002/000024248.jpg", "caption":"hallo", "topic":"d"},
        {"img_id":14, "image":"/topical_data/output_images/output_images/sublist_6/00002/000024248.jpg", "caption":"hallo", "topic":"d"},
        {"img_id":15, "image":"/topical_data/output_images/output_images/sublist_6/00002/000024248.jpg", "caption":"hallo", "topic":"d"},
        {"img_id":16, "image":"/topical_data/output_images/output_images/sublist_6/00002/000024248.jpg", "caption":"hallo", "topic":"d"},
        {"img_id":17, "image":"/topical_data/output_images/output_images/sublist_6/00002/000024248.jpg", "caption":"hallo", "topic":"d"},
        {"img_id":18, "image":"/topical_data/output_images/output_images/sublist_6/00002/000024248.jpg", "caption":"hallo", "topic":"d"},
        
    ]

    df = pd.DataFrame(data)
    sampler = NegativeSampler(df, 2, "diff_class")

    print(sampler.sample(2))
