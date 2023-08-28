from config import *

# Padding theo max_length
def pad_to_len(list_data, max_len, pad_value):
    list_data = list_data[-max_len:]
    len_to_pad = max_len - len(list_data)
    pads = [pad_value] * len_to_pad
    list_data.extend(pads)
    return list_data

# Lấy tập từ vựng từ Emotion từ dữ liệu
def get_vocabs(file_paths):
    emotion_vocab = vocab.Vocab()
    # keep 'neutral' in index-0
    emotion_vocab.word2index('neutral', train=True)
    # global speaker_vocab, emotion_vocab
    for file_path in file_paths:
        data = pd.read_csv(file_path)
        for row in tqdm(data.iterrows(),
                        desc='get vocab from {}'.format(file_path), disable=CONFIG['local_rank'] not in [-1, 0]):
            meta = row[1]
            emotion = meta['Emotion'].lower()
            emotion_vocab.word2index(emotion, train=True)
    if CONFIG['local_rank'] in [-1, 0]:
        torch.save(emotion_vocab.to_dict(), CONFIG['emotion_vocab'])
    logging.info('total {} emotions'.format(len(emotion_vocab)))

# Tạo dataset
def build_dataset(dialogues, train=False):
    ret_utterances = []
    ret_labels = []
    for dialogue in dialogues:
        utterance_ids = []
        query = 'Với câu:'
        query_ids = tokenizer(query)['input_ids'][1:-1] # word id in query bỏ CLS và SEP
        for idx, turn_data in enumerate(dialogue): # Duyệt qua hội thoại
            text_with_speaker = f" {turn_data['speaker']}" + ' : ' + f"{turn_data['text']}" # Ghép tên người nói với lời thoại
            token_ids = tokenizer(text_with_speaker)['input_ids'][1:] # Tokenize câu vừa ghép
            utterance_ids.append(token_ids) # Lưu token_id vào mảng
            if turn_data['label'] < 0: # Nếu nhãn <0 (rỗng) -> bỏ qua
                continue
            full_context = [CONFIG['CLS']] 
            lidx = 0
            for lidx in range(idx):
                total_len = sum([len(item) for item in utterance_ids[lidx:]]) + 8
                if total_len + len(utterance_ids[idx]) <= CONFIG['max_len']:
                    break
            lidx = max(lidx, idx-8)
            for item in utterance_ids[lidx:]:
                full_context.extend(item)

            query_idx = idx
            prompt = f"{dialogue[query_idx]['speaker']}" + ' cảm thấy <mask>'
            full_query = query_ids + utterance_ids[query_idx] + tokenizer(prompt)['input_ids'][1:]
            input_ids = full_context + full_query
            input_ids = pad_to_len(input_ids, CONFIG['max_len'], CONFIG['pad_value'])
            ret_utterances.append(input_ids)
            ret_labels.append(dialogue[query_idx]['label'])

            if train and idx > 3 and torch.rand(1).item() < 0.2:
                query_idx = random.randint(lidx, idx-1)
                if dialogue[query_idx]['label'] < 0:
                    continue
                prompt = f" {dialogue[query_idx]['speaker']}" + ' cảm thấy <mask>'
                full_query = query_ids + utterance_ids[query_idx] + tokenizer(prompt)['input_ids'][1:]
                input_ids = full_context + full_query
                input_ids = pad_to_len(input_ids, CONFIG['max_len'], CONFIG['pad_value'])
                ret_utterances.append(input_ids)
                ret_labels.append(dialogue[query_idx]['label'])
    dataset = TensorDataset(
        torch.LongTensor(ret_utterances),
        torch.LongTensor(ret_labels)
    )
    return dataset

def load_turn(file_path):
    emotion_vocab = vocab.Vocab.from_dict(torch.load(CONFIG['emotion_vocab']))
    data = pd.read_csv(file_path)
    pre_dial_id = -1
    dialogues = []
    dialogue = []
    speaker_vocab = vocab.Vocab()
    for row in tqdm(data.iterrows(),
                    desc='processing file {}'.format(file_path), disable=CONFIG['local_rank'] not in [-1, 0]):
        meta = row[1]
        text = meta['Utterance_clean']
        speaker = meta['Id_speaker']
        emotion = meta['Emotion'].lower()
        emotion_idx = emotion_vocab.word2index(emotion)
        turn_data = {}
        turn_data['speaker'] = speaker
        speaker_vocab.word2index(speaker, train=True)
        turn_data['text'] = text
        turn_data['label'] = emotion_idx

        dialogue_id = meta['Dialog_id']
        if pre_dial_id == -1:
            pre_dial_id = dialogue_id
        if dialogue_id != pre_dial_id:
            dialogues.append(dialogue)
            dialogue = []
        pre_dial_id = dialogue_id
        dialogue.append(turn_data)
    dialogues.append(dialogue)
    # speaker_vocab = speaker_vocab.prune_by_count(30)
    # for speaker_name in speaker_vocab.counts.keys():
    #     tokenizer.add_tokens(speaker_name)
    return dialogues



# Tạo dataset
def build_dataset_utters(dialogues, train=False):
    ret_utterances = []
    ret_labels = []
    for dialogue in dialogues:
        utterance_ids = []
        query = 'Với câu:'
        query_ids = tokenizer(query)['input_ids'][1:-1] # word id in query bỏ CLS và SEP
        for idx, turn_data in enumerate(dialogue): # Duyệt qua hội thoại
            text_with_speaker = f" {turn_data['speaker']}" + ' : ' + f"{turn_data['text']}" # Ghép tên người nói với lời thoại
            token_ids = tokenizer(text_with_speaker)['input_ids'][1:] # Tokenize câu vừa ghép
            utterance_ids.append(token_ids) # Lưu token_id vào mảng
            if turn_data['label'] < 0: # Nếu nhãn <0 (rỗng) -> bỏ qua
                continue
            full_context = [CONFIG['CLS']] 
            lidx = 0
            for lidx in range(idx):
                total_len = sum([len(item) for item in utterance_ids[lidx:]]) + 8
                if total_len + len(utterance_ids[idx]) <= CONFIG['max_len']:
                    break
            lidx = max(lidx, idx-8)
            for item in utterance_ids[lidx:]:
                full_context.extend(item)

            query_idx = idx
            prompt = f"{dialogue[query_idx]['speaker']}" + ' cảm thấy <mask>'
            full_query = query_ids + utterance_ids[query_idx] + tokenizer(prompt)['input_ids'][1:]
            input_ids = full_context + full_query
            input_ids = pad_to_len(input_ids, CONFIG['max_len'], CONFIG['pad_value'])
            ret_utterances.append(input_ids)
            ret_labels.append(dialogue[query_idx]['label'])

            if train and idx > 3 and torch.rand(1).item() < 0.2:
                query_idx = random.randint(lidx, idx-1)
                if dialogue[query_idx]['label'] < 0:
                    continue
                prompt = f" {dialogue[query_idx]['speaker']}" + ' cảm thấy <mask>'
                full_query = query_ids + utterance_ids[query_idx] + tokenizer(prompt)['input_ids'][1:]
                input_ids = full_context + full_query
                input_ids = pad_to_len(input_ids, CONFIG['max_len'], CONFIG['pad_value'])
                ret_utterances.append(input_ids)
                ret_labels.append(dialogue[query_idx]['label'])
    dataset = TensorDataset(
        torch.LongTensor(ret_utterances),
        torch.LongTensor(ret_labels)
    )
    return dataset

def load_turn_utters(file_path):
    emotion_vocab = vocab.Vocab.from_dict(torch.load(CONFIG['emotion_vocab']))
    data = pd.read_csv(file_path)
    dialogues = []
    speaker_vocab = vocab.Vocab()
    for row in tqdm(data.iterrows(),
                    desc='processing file {}'.format(file_path), disable=CONFIG['local_rank'] not in [-1, 0]):
        dialogue = []
        meta = row[1]
        text = meta['Utterance_clean']
        speaker = meta['Id_speaker']
        emotion = meta['Emotion'].lower()
        emotion_idx = emotion_vocab.word2index(emotion)
        turn_data = {}
        turn_data['speaker'] = speaker
        speaker_vocab.word2index(speaker, train=True)
        turn_data['text'] = text
        turn_data['label'] = emotion_idx

        dialogue.append(turn_data)
        dialogues.append(dialogue)
    # speaker_vocab = speaker_vocab.prune_by_count(30)
    # for speaker_name in speaker_vocab.counts.keys():
    #     tokenizer.add_tokens(speaker_name)
    return dialogues