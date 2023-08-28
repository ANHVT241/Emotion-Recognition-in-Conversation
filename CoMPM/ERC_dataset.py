from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence
import random
import pandas as pd

# Data_loader
class Data_loader(Dataset):
    def __init__(self, csv_file_path, model_path):
        self.dialogs = []

        # Read_file
        df = pd.read_csv(csv_file_path)
#         df = df.dropna()

        temp_speakerList = []
        context = []
        context_speaker = []
        self.speakerNum = []

        # Format lại nhãn
        self.emoSet = set()

        # id hội thoại đầu tiên
        dialog_id = df['Dialog_id'][0]

        # Iterate
        for i, data in df.iterrows():

            # Check Kết thúc 1 hội thoại
            if data['Dialog_id'] != dialog_id and len(self.dialogs) > 0:
                self.speakerNum.append(len(temp_speakerList))
                temp_speakerList = []
                context = []
                context_speaker = []
                dialog_id = data['Dialog_id']

            
            speaker, utt, emo = data['Id_speaker'], data['Utterance_clean'], data['Emotion']

            context.append(utt)
            
            # Check speaker mới
            if speaker not in temp_speakerList:
                temp_speakerList.append(speaker)
            speakerCLS = temp_speakerList.index(speaker)
            context_speaker.append(speakerCLS)
    
            self.dialogs.append([context_speaker[:], context[:], emo])
            self.emoSet.add(emo)

        self.emoList = sorted(self.emoSet)
        self.labelList = self.emoList
        self.speakerNum.append(len(temp_speakerList))

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, idx):
        return self.dialogs[idx], self.labelList