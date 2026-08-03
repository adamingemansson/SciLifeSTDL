"""Pure released-STPath fine-tuning benchmark for Gen6 arm A."""
from __future__ import annotations

import torch
import torch.nn as nn


class FineTunedSTPathBenchmark(nn.Module):
    """Use STPath's own prediction head directly, with no Gen3 extras.

    Query expression is always masked by STPath's tokenizer.  Query H&E is
    represented as unavailable, matching the spatial-hole task rather than
    leaking held-out patches into the benchmark.
    """
    def __init__(self, *, gene_names: list[str], gene_vocab_path: str,
                 checkpoint_path: str, default_organ: str = "Kidney"):
        super().__init__()
        from src.models.stpath_encoder import STPathContextEncoder

        self.gene_names = list(gene_names)
        self.encoder = STPathContextEncoder(
            gene_names=self.gene_names, gene_voc_path=gene_vocab_path,
            model_weight_path=checkpoint_path, organ_type=default_organ,
            tech_type="Visium", hidden_dim=256, pretrained=True,
            input_already_log1p=True,
        )
        # Load the released checkpoint first, then explicitly unfreeze it.
        # Flipping `pretrained` also selects STPathContextEncoder's live
        # autograd path instead of its frozen no_grad inference path.
        self.encoder.pretrained = False
        for parameter in self.encoder.model.parameters():
            parameter.requires_grad_(True)
        # The pure benchmark returns STPath's official prediction head;
        # these downstream projection layers are intentionally bypassed.
        for module in (self.encoder.embedding_norm, self.encoder.proj):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def forward(self, inputs) -> dict:
        device = next(self.parameters()).device
        context_coords = torch.as_tensor(inputs.observed_coords, dtype=torch.float32, device=device)
        query_coords = torch.as_tensor(inputs.query_coords, dtype=torch.float32, device=device)
        context_expression = torch.as_tensor(
            inputs.observed_full_gene_expression, dtype=torch.float32, device=device,
        )
        context_images = torch.as_tensor(
            inputs.observed_gigapath_features, dtype=torch.float32, device=device,
        )
        context_available = torch.as_tensor(
            inputs.observed_image_available, dtype=torch.bool, device=device,
        )
        query_images = torch.zeros(
            query_coords.shape[0], context_images.shape[1], dtype=context_images.dtype, device=device,
        )
        query_available = torch.zeros(query_coords.shape[0], dtype=torch.bool, device=device)
        organ = getattr(inputs, "sample_organ", None)
        if organ:
            self.encoder.organ_type = str(organ)
        supported = self.encoder(
            context_coords, context_expression, query_coords,
            context_images, query_images,
            context_image_available=context_available,
            query_image_available=query_available,
            return_official_predictions=True,
        )
        expression = torch.zeros(
            query_coords.shape[0], len(self.gene_names), dtype=supported.dtype, device=device,
        )
        valid_positions = torch.as_tensor(
            self.encoder._valid_gene_pos, dtype=torch.long, device=device,
        )
        expression[:, valid_positions] = supported
        return {"expression": expression, "query_hidden": supported}
