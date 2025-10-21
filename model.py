import torch.nn as nn
import torch


class Model(nn.Module):   
    def __init__(self, encoder , config ):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config

        # Hidden size resolution (keep your fallback order)
        self.hidden_size = getattr(config, "hidden_size", None)
        if self.hidden_size is None:
            self.hidden_size = getattr(config, "d_model", None)

        # Use inner encoder for CodeT5, like your earlier versions
        if "codet5" in getattr(config, "_name_or_path", "").lower():
            self.encoder = encoder.encoder

        # Always define the head to avoid attribute errors
        self.classification_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(self.hidden_size // 2, 1),
            nn.Sigmoid()
        )

        # Cache pad id (CodeT5 pads with 0); fallback to 1 if missing
        self.pad_id = getattr(config, "pad_token_id", 0 if "codet5" in getattr(config, "_name_or_path", "").lower() else 1)

    def forward(self, code_inputs=None, nl_inputs=None):
        # Normalize tasks into a string for a robust check
        tasks = getattr(self.config, "tasks", None)
        if isinstance(tasks, (list, tuple)):
            task_str = " ".join(map(str, tasks)).lower()
        else:
            task_str = str(tasks).lower() if tasks is not None else ""
        is_code_search = ("code_search" in task_str) or ("embedding" in task_str)  # treat either as embedding mode

        if code_inputs is not None:
            if "codet5" in self.config._name_or_path.lower():
                outputs = self.encoder(code_inputs, attention_mask=code_inputs.ne(self.pad_id))
                outputs = outputs.last_hidden_state
                attention_mask = code_inputs.ne(self.pad_id)
                mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.size()).float()
                sum_embeddings = torch.sum(outputs * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                outputs = sum_embeddings / sum_mask
            else:
                outputs = self.encoder(code_inputs, attention_mask=code_inputs.ne(self.pad_id))[1]

            if is_code_search:
                return torch.nn.functional.normalize(outputs, p=2, dim=1).squeeze()
            else:
                return self.classification_head(outputs).squeeze()

        else:  # NL branch 
            if "codet5" in self.config._name_or_path.lower():
                outputs = self.encoder(nl_inputs, attention_mask=nl_inputs.ne(self.pad_id))
                outputs = outputs.last_hidden_state
                attention_mask = nl_inputs.ne(self.pad_id)
                mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.size()).float()
                sum_embeddings = torch.sum(outputs * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                outputs = sum_embeddings / sum_mask
            else:
                outputs = self.encoder(nl_inputs, attention_mask=nl_inputs.ne(self.pad_id))[1]

            if is_code_search:
                return torch.nn.functional.normalize(outputs, p=2, dim=1).squeeze()
            else:
                return self.classification_head(outputs).squeeze()