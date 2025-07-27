import torch
from torch.utils.data import Dataset, DataLoader

class ChunkDataset(Dataset):
    def __init__(self, df):
        super(ChunkDataset, self).__init__()

        self.df = df
        self.pat_list = list(set(self.df["patient"]))
      
        
    def __len__(self):
        return len(list(set(self.df["patient"])))

    def __getitem__(self, i):
        pat = self.pat_list[i]
  
        patient_list = list((self.df['patient']))
        idx = patient_list.index(pat)
  
        emblist = []
        while idx<len(patient_list) and (self.df['patient'][idx]) == pat:
            emblist.append(self.df["emb"][idx])
            if "emb1" in self.df.columns:
                emblist.append(self.df["emb1"][idx])
            idx+=1
        x =torch.tensor(emblist) 
        M = 256
        D = x.shape[1]
        S = x.shape[0]
        idx = patient_list.index(pat)
        
        padded = torch.zeros(
            M, D,
            dtype=x.dtype,
            device=x.device
        )
        
        
        padded[:S] = x
     
        return padded, int(self.df["label"][idx]), int(pat)

