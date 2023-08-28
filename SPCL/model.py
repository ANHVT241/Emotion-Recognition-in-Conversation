import torch
import torch.nn as nn
from transformers import AutoModel
from sklearn.metrics import euclidean_distances
import torch.nn.functional as F

class CLModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Khởi tạo các thuộc tính dựa trên config
        self.dropout = config['dropout']
        self.num_classes = config['num_classes']
        self.pad_value = config['pad_value']
        self.mask_value = config['mask_value']

        # Tải một mô hình BERT
        self.f_context_encoder = AutoModel.from_pretrained(config['bert_path'], local_files_only=False)

        # Lấy số word embeddings và kích thước của chúng từ mô hình BERT
        num_embeddings, self.dim = self.f_context_encoder.embeddings.word_embeddings.weight.data.shape

        # Thay đổi kích thước của token embeddings để chứa các token bổ sung
        self.f_context_encoder.resize_token_embeddings(num_embeddings + 256)

        # Định nghĩa một layer tuyến tính cho việc dự đoán
        self.predictor = nn.Sequential(
            nn.Linear(self.dim, self.num_classes)
        )

        # Định nghĩa một layer tuyến tính 'g' cho việc biến đổi MLP (nếu được chỉ định trong config)
        self.g = nn.Sequential(
            nn.Linear(self.dim, self.dim),
        )

    def device(self):
        # Trả về thiết bị (cpu/gpu)
        return self.f_context_encoder.device

    def gen_f_reps(self, sentences):
        '''
        Tạo các biểu diễn vector cho mỗi lượt trò chuyện
        '''
        batch_size, max_len = sentences.shape[0], sentences.shape[-1]

        # Thay đổi hình dạng của các câu để tiến hành xử lý
        sentences = sentences.reshape(-1, max_len)

        # Tạo mặt nạ cho các giá trị pad
        mask = 1 - (sentences == (self.pad_value)).long()

        # Mã hóa từng câu sử dụng mô hình BERT
        utterance_encoded = self.f_context_encoder(
            input_ids=sentences,
            attention_mask=mask,
            output_hidden_states=True,
            return_dict=True
        )['last_hidden_state']

        # Tìm vị trí của các giá trị được mask
        mask_pos = (sentences == (self.mask_value)).long().max(1)[1]

        # Trích xuất các đầu ra đã mã hóa tương ứng
        mask_outputs = utterance_encoded[torch.arange(mask_pos.shape[0]), mask_pos, :]

        feature = mask_outputs

        # Nếu output là mlp thì cho qua 1 lớp MLP
        if self.config['output_mlp']:
            feature = self.g(feature)

        return feature

    def forward(self, reps, centers, score_func):
        num_classes, num_centers = centers.shape[0], centers.shape[1]

        # Mở rộng kích thước của các representations và centers để thực hiện các phép tính ma trận
        reps = reps.unsqueeze(1).expand(reps.shape[0], num_centers, -1)
        reps = reps.unsqueeze(1).expand(reps.shape[0], num_classes, num_centers, -1)

        centers = centers.unsqueeze(0).expand(reps.shape[0], -1, -1, -1)

        # Tính ma trận tương đồng sử dụng hàm tính điểm được cung cấp
        sim_matrix = score_func(reps, centers)

        # Các điểm số kết quả
        scores = sim_matrix
        return scores

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}
        self.emb_name = 'word_embeddings.weight'

    def attack(self, epsilon=1.):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
    
    def save_checkpoint(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
    def load_checkpoint(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
    
    