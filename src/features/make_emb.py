import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
from src import BuildDataset
from torch.utils.data import DataLoader

tqdm.pandas()


class MakeEmbeddings:
    def __init__(
        self,
        df: DataFrame,
        batch_size: int,
        device: str,
        name_model: str,
    ):
        self.df = df
        self.batch_size = batch_size
        self.device = device
        self.name_model = name_model
        self.list_emb = []

    def __export_tokenazer_model(self, name_model):
        self.tokenizer = AutoTokenizer.from_pretrained(name_model)
        self.model = AutoModel.from_pretrained(name_model).to(self.device)

    def __get_embedded(self, train_load):
        for batch in tqdm(train_load):
            with torch.no_grad():
                token = torch.squeeze(batch["input_ids"], dim=1).to(self.device)
                mask = torch.squeeze(batch["attention_mask"], dim=1).to(self.device)
                batch_embeddings = self.model(token, attention_mask=mask)
                self.list_emb.append(batch_embeddings[0][:, 0, :].cpu().numpy())

    def run(self):
        print(f"get embeddings with {self.name_model}")
        self.__export_tokenazer_model(self.name_model)
        data = BuildDataset(self.df, self.tokenizer)
        train_load = DataLoader(data, batch_size=self.batch_size, num_workers=0)
        self.__get_embedded(train_load)

        embed_np = np.concatenate(self.list_emb)
        self.list_emb = []
        df_emb_text = pd.DataFrame(embed_np)
        df_emb_text["target"] = self.df["level"]
        return df_emb_text
