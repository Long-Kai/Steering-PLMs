import torch
from typing import Union, Tuple, List
from attr import dataclass
from esm3.utils.constants import esm3 as C
from esm3.utils.structure.affine3d import build_affine3d_from_coordinates, Affine3D

@dataclass
class ESMOutput:
    sequence_logits: torch.Tensor
    structure_logits: torch.Tensor
    secondary_structure_logits: torch.Tensor
    sasa_logits: torch.Tensor
    function_logits: torch.Tensor
    residue_logits: torch.Tensor
    embeddings: torch.Tensor

def esm3_steering_forward(
        self,
        *,
        sequence_tokens: Union[torch.Tensor, None] = None,
        structure_tokens: Union[torch.Tensor, None] = None,
        ss8_tokens: Union[torch.Tensor, None] = None,
        sasa_tokens: Union[torch.Tensor, None] = None,
        function_tokens: Union[torch.Tensor, None] = None,
        residue_annotation_tokens: Union[torch.Tensor, None] = None,
        average_plddt: Union[torch.Tensor, None] = None,
        per_res_plddt: Union[torch.Tensor, None] = None,
        structure_coords: Union[torch.Tensor, None] = None,
        chain_id: Union[torch.Tensor, None] = None,
        sequence_id: Union[torch.Tensor, None] = None,
        steering_vectors: Union[List[torch.Tensor], None] = None, 
    ) -> ESMOutput:
        """
        Performs Steering forward pass through the ESM3 model. 

        Args:
            sequence_tokens (torch.Tensor, optional): The amino acid tokens.
            structure_tokens (torch.Tensor, optional): The structure tokens.
            ss8_tokens (torch.Tensor, optional): The secondary structure tokens.
            sasa_tokens (torch.Tensor, optional): The solvent accessible surface area tokens.
            function_tokens (torch.Tensor, optional): The function tokens.
            residue_annotation_tokens (torch.Tensor, optional): The residue annotation tokens.
            average_plddt (torch.Tensor, optional): The average plddt across the entire sequence.
            per_res_plddt (torch.Tensor, optional): The per residue plddt, if you want to specify exact plddts, use this,
                otherwise, use average_plddt.
            structure_coords (torch.Tensor, optional): The structure coordinates, in the form of (B, L, 3, 3).
            chain_id (torch.Tensor, optional): The chain ID
            sequence_id (torch.Tensor, optional): The sequence ID.

        Returns:
            ESMOutput: The output of the ESM3 model.

        Raises:
            ValueError: If at least one of the inputs is None.

        """
        # Reasonable defaults:
        try:
            L, device = next(
                (x.shape[1], x.device)
                for x in [
                    sequence_tokens,
                    structure_tokens,
                    ss8_tokens,
                    sasa_tokens,
                    structure_coords,
                    function_tokens,
                    residue_annotation_tokens,
                ]
                if x is not None
            )
        except StopIteration:
            raise ValueError("At least one of the inputs must be non-None")

        t = self.tokenizers
        defaults = lambda x, tok: (
            torch.full((1, L), tok, dtype=torch.long, device=device) if x is None else x
        )
        sequence_tokens = defaults(sequence_tokens, t.sequence.mask_token_id)
        ss8_tokens = defaults(ss8_tokens, C.SS8_PAD_TOKEN)
        sasa_tokens = defaults(sasa_tokens, C.SASA_PAD_TOKEN)
        average_plddt = defaults(average_plddt, 1).float()
        per_res_plddt = defaults(per_res_plddt, 0).float()
        chain_id = defaults(chain_id, 0)

        if residue_annotation_tokens is None:
            residue_annotation_tokens = torch.full(
                (1, L, 16), C.RESIDUE_PAD_TOKEN, dtype=torch.long, device=device
            )

        if function_tokens is None:
            function_tokens = torch.full(
                (1, L, 8), C.INTERPRO_PAD_TOKEN, dtype=torch.long, device=device
            )

        if structure_coords is None:
            structure_coords = torch.full(
                (1, L, 3, 3), float("nan"), dtype=torch.float, device=device
            )

        structure_coords = structure_coords[
            ..., :3, :
        ]  # In case we pass in an atom14 or atom37 repr
        affine, affine_mask = build_affine3d_from_coordinates(structure_coords)

        structure_tokens = defaults(structure_tokens, C.STRUCTURE_MASK_TOKEN)
        assert structure_tokens is not None
        structure_tokens = (
            structure_tokens.masked_fill(structure_tokens == -1, C.STRUCTURE_MASK_TOKEN)
            .masked_fill(sequence_tokens == C.SEQUENCE_BOS_TOKEN, C.STRUCTURE_BOS_TOKEN)
            .masked_fill(sequence_tokens == C.SEQUENCE_PAD_TOKEN, C.STRUCTURE_PAD_TOKEN)
            .masked_fill(sequence_tokens == C.SEQUENCE_EOS_TOKEN, C.STRUCTURE_EOS_TOKEN)
            .masked_fill(
                sequence_tokens == C.SEQUENCE_CHAINBREAK_TOKEN,
                C.STRUCTURE_CHAINBREAK_TOKEN,
            )
        )

        x = self.encoder(
            sequence_tokens,
            structure_tokens,
            average_plddt,
            per_res_plddt,
            ss8_tokens,
            sasa_tokens,
            function_tokens,
            residue_annotation_tokens,
        )
        x, embedding = self.transformer.steering_forward(x, sequence_id, affine, affine_mask, chain_id, steering_vectors=steering_vectors)
        return self.output_heads(x, embedding)


def steering_forward(
    self,
    x: torch.Tensor,
    sequence_id: Union[torch.Tensor, None] = None,
    affine: Union[Affine3D, None] = None,
    affine_mask: Union[torch.Tensor, None] = None,
    chain_id: Union[torch.Tensor, None] = None,
    steering_vectors: Union[List[torch.Tensor], None] = None, 
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Steering Forward pass of the TransformerStack.

    Args:
        x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, d_model).
        sequence_id (torch.Tensor): The sequence ID tensor of shape (batch_size, sequence_length).
        affine (Affine3D, None]): The affine transformation tensor or None.
        affine_mask (torch.Tensor, None]): The affine mask tensor or None.
        chain_id (torch.Tensor): The protein chain tensor of shape (batch_size, sequence_length).
            Only used in geometric attention.

    Returns:
        post_norm: The output tensor of shape (batch_size, sequence_length, d_model).
        pre_norm: The embedding of shape (batch_size, sequence_length, d_model).
    """
    *batch_dims, _ = x.shape
    if chain_id is None:
        chain_id = torch.ones(size=batch_dims, dtype=torch.int64, device=x.device)
    
    for l, block in enumerate(self.blocks):
        x = block(x, sequence_id, affine, affine_mask, chain_id)

        if steering_vectors is not None:
            add_x = steering_vectors[l]
            new_x = x + add_x
            new_x_norm = torch.norm(new_x, p=2, dim=-1, keepdim=True).detach()
            x_norm = torch.norm(x, p=2, dim=-1, keepdim=True).detach()
            x = new_x * (x_norm / new_x_norm) 

    return self.norm(x), x

