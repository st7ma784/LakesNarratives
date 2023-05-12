from torchvision import transforms
from PIL import Image
import torch     
import os
import zipfile
from pySmartDL import SmartDL
import pytorch_lightning as pl
from transformers import AutoTokenizer,GPT2Tokenizer, CLIPTokenizer
import time
from pathlib import Path

class NarrativesDataModule(pl.LightningDataModule):

    def __init__(self, Cache_dir='.', batch_size=256):
        super().__init__()
        self.data_dir = Cache_dir
        self.ann_dir=os.path.join(self.data_dir,"annotations")
        self.batch_size = batch_size
        self.splits={"train":[],"val":[],"test":[]}
        self.tokenizer=CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32",cache_dir=self.data_dir)
        # try: 
        #     self.tokenizer=AutoTokenizer.from_pretrained("gpt2",cache_dir=self.data_dir)
        # except ValueError as e:
        #     from transformers import GPT2Tokenizer
        #     tok = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=self.data_dir)
        #     tok.save_pretrained(self.data_dir)
        # finally:
        #     self.tokenizer=AutoTokenizer.from_pretrained("gpt2",cache_dir=self.data_dir)
        #self.tokenizer.vocab["</s>"] = self.tokenizer.vocab_size -1
        #self.tokenizer.pad_token = self.tokenizer.eos_token 
        self.prepare_data()
    def train_dataloader(self, B=None):
        if B is None:
            B=self.batch_size 
        return torch.utils.data.DataLoader(self.train, batch_size=B, shuffle=True, num_workers=2, prefetch_factor=2, pin_memory=False,drop_last=True)
    def val_dataloader(self, B=None):
        if B is None:
            B=self.batch_size
       
        return torch.utils.data.DataLoader(self.val, batch_size=B, shuffle=True, num_workers=2, prefetch_factor=2, pin_memory=True,drop_last=True)
    def test_dataloader(self,B=None):
        if B is None:
            B=self.batch_size


        return torch.utils.data.DataLoader(self.test, batch_size=B, shuffle=True, num_workers=4, prefetch_factor=4, pin_memory=True,drop_last=True)
    def prepare_data(self):
        #Downlaod LakesNarratives dataset
        datadir=os.path.join(self.data_dir,"data")
        #URL is github.com/SpaceTimeNarratives/data.zip
        if not os.path.exists(datadir):
            os.makedirs(datadir)
        #print("Downloading data")
        if not os.path.exists(os.path.join(datadir,"data.zip")):

            url = "github.com/SpaceTimeNarratives/data.zip"
            dest = os.path.join(datadir,"data.zip")
            obj = SmartDL(url, dest, progress_bar=True)
            obj.start()
            #print("Downloaded data")
            #print("Extracting data")
            with zipfile.ZipFile(dest, 'r') as zip_ref:
                zip_ref.extractall(datadir)

        


    def setup(self, stage=None):
        '''called on each GPU separately - stage defines if we are at fit or test step'''
        #print("Entered COCO datasetup")
        #assume data is downloaded and extracted
        #files are +emotion.json
        # data.xlsx
        # -emotion.json
        # geonoun.json
        # locadv.json
        # plname.json
        # sp-prep.json


        #import library to read excel file
        import pandas as pd
        #read excel file
        df = pd.read_excel(os.path.join(self.data_dir,"data.xlsx"))
        #create torch dataset

        dataset = NarrativesDataset(df,self.tokenizer)
        #split dataset into train, val, test
        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        self.train, self.val, self.test = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

from torch.utils.data import Dataset

class NarrativesDataset(Dataset):
    def __init__(self, df, tokenizer) -> None:
        super().__init__()
        self.df = df
        self.tokenizer = tokenizer
        '''
        Tokenize the text column. then every item in each other field
        '''
        #print df columns
        print(self.df.columns) 
        self.df["text"]=self.df["text"].apply(lambda x: self.tokenizer(x,return_tensors="pt",padding="max_length", truncation=True,max_length=128))
        #self.df["plnames"]=self.df["plnames"].apply(lambda x: self.tokenizer(x,return_tensors="pt",padding=True, truncation=True,max_length=512))
        self.log={}
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        #return a dictionary of text and one hot masks
        entry=self.df.iloc[index] 
        text=entry["text"] #already tokenized
        #as tensor
        text=text["input_ids"]
        returndict={"text":text}

        for name in ['plnames', 'geonouns',  'pos_emotions','neg_emotions', 'loc_advs', 'sp_prep']:
            #print(name)
            #get location data from self.df
            column=entry[name] # this is a string of the form "[(name, start, end), (name, start, end), ...]" but name is a string that may contain spaces and other characters
            #clean up the string by removing name and converting the start and end to indexes in text
            column=column.replace("(","").replace(")","").replace("[","").replace("]","").replace("'","").replace(" ","").split(",")
            #empty list to often is [""]
            if column==[""]:
                column=[]
            mask=torch.zeros_like(text)
            #for every third element in the list, tokenize the element. 
                #the first element is the name, the second is the start index, the third is the end index
            for element in range(0,len(column),3):
                #print("element",element)
                if not isinstance(column[element],str):
                    print("not string", column[element])
                tokenizedElement=self.tokenizer(column[element],return_tensors="pt",padding=False, truncation=True,max_length=512)
                #may be multiple tokens
                #keep log of places in dictionary
                self.log.update({name:tokenizedElement})
                tokenizedElement=tokenizedElement["input_ids"][0]
                for tokenidx in range(1,len(tokenizedElement)-1):
                    #print("tokenidx",tokenidx)
                    #print("tokenizedElement[tokenidx]",tokenizedElement[tokenidx])
                    mask=torch.where(text==tokenizedElement[tokenidx],tokenizedElement[tokenidx],mask)
            returndict[name]=mask
        return returndict


        


if __name__ == "__main__":
    #run this to test the dataloader
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--Cache_dir', type=str, default='.', help='path to download and cache data')
    dir=os.path.join(parser.parse_args().Cache_dir,"data")
    dm = NarrativesDataModule(dir)
    dm.prepare_data()
    dm.setup()
    train_loader = dm.train_dataloader()
    for batch in train_loader:
        print(batch)
        break