"""Paridade bit a bit entre os dois storages do K2 Training Base.

O contrato numérico do node é: pesos no grid fp8 CRU, GEMM em bf16
(comfy_cast_weights). A mudança de 2026-08-08 troca o storage de bf16
materializado (2 bytes/peso — dobrava o modelo e derrubava máquinas fracas)
por fp8 puro (1 byte/peso) com cast fp8->bf16 por camada no forward.

Estes testes provam, em CPU e sem depender do ComfyUI, que a troca é
algebricamente nula: o cast fp8->bf16 é exato (todo ponto do grid e4m3fn é
representável em bf16), logo o peso que entra no F.linear é bit-idêntico nos
dois modos — inclusive depois da fusão da turbo.
"""

import unittest

import torch
import torch.nn.functional as F


def _grid(w):
    """O passo do trainer: re-quantiza no grid fp8 cru (base.py:547)."""
    return w.to(torch.float8_e4m3fn)


class TrainbaseStorageParityTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.w = (torch.randn(96, 64) * 0.02).to(torch.bfloat16)

    def test_fp8_to_bf16_cast_is_exact(self):
        """Todo valor e4m3fn é representável em bf16 (mantissa 3 <= 8 bits,
        expoente coberto): fp8->bf16->fp8 é a identidade."""
        todos = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn)
        finitos = todos[~torch.isnan(todos.float())]
        ida_e_volta = finitos.to(torch.bfloat16).to(torch.float8_e4m3fn)
        self.assertTrue(torch.equal(ida_e_volta.view(torch.uint8),
                                    finitos.view(torch.uint8)))

    def test_storage_fp8_equals_storage_bf16_no_forward(self):
        # antigo: materializa bf16; novo: guarda fp8 e casta no forward
        legado = _grid(self.w).to(torch.bfloat16)
        novo = _grid(self.w)                      # 1 byte/peso em repouso
        self.assertEqual(novo.element_size(), 1)
        self.assertTrue(torch.equal(novo.to(torch.bfloat16), legado))

        x = torch.randn(4, 64).to(torch.bfloat16)
        y_legado = F.linear(x, legado)
        y_novo = F.linear(x, novo.to(torch.bfloat16))
        self.assertTrue(torch.equal(y_novo, y_legado))

    def test_turbo_merge_parity(self):
        """W = fp8( fp8(W_deq) + s*up@down ): mesmo resultado nos dois storages."""
        torch.manual_seed(1)
        up = torch.randn(96, 8) * 0.05
        down = torch.randn(8, 64) * 0.05
        s = 1.0

        # caminho legado: base bf16 (ja no grid) -> fp32 -> soma -> fp8 -> bf16
        base_legado = _grid(self.w).to(torch.bfloat16)
        delta = (up @ down).to(torch.float32)
        merged_legado = (base_legado.to(torch.float32) + s * delta)
        merged_legado = merged_legado.to(torch.float8_e4m3fn).to(torch.bfloat16)

        # caminho novo: base fp8 -> fp32 -> soma -> fp8 (fica em 1 byte)
        base_novo = _grid(self.w)
        merged_novo = (base_novo.to(torch.float32) + s * delta)
        merged_novo = merged_novo.to(torch.float8_e4m3fn)

        self.assertEqual(merged_novo.element_size(), 1)
        self.assertTrue(torch.equal(merged_novo.to(torch.bfloat16), merged_legado))

    def test_lora_bake_requant_matches_runner_convention(self):
        """`set_weight` com layout_type=None faz weight.to(fp8): o delta de um
        LoRA assado re-quantiza no grid CRU, como o runner faz com deltas."""
        base = _grid(self.w)
        delta = torch.randn_like(self.w, dtype=torch.float32) * 0.01
        assado = (base.to(torch.float32) + delta).to(base.dtype)  # o .to() do set_weight
        self.assertEqual(assado.dtype, torch.float8_e4m3fn)


if __name__ == '__main__':
    unittest.main()
