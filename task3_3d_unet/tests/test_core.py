import unittest

import torch

from task3_3d_unet.common import FIELDS, MODALITIES, Task3Error, directed_pairs, joint_domain
from task3_3d_unet.losses import (
    block_mask,
    dilated_foreground,
    fine_tuning_loss,
    masked_reconstruction_loss,
    normalized_l1,
)
from task3_3d_unet.model import FiLMResidualBlock, ModelConfig, build_model


class DomainTests(unittest.TestCase):
    def test_fifteen_domains_and_twenty_directed_pairs(self):
        domains = {joint_domain(modality, field) for modality in MODALITIES for field in FIELDS}
        self.assertEqual(domains, set(range(15)))
        self.assertEqual(len(directed_pairs()), 20)
        self.assertTrue(all(source != target for source, target in directed_pairs()))

    def test_default_architecture_contract(self):
        config = ModelConfig()
        self.assertEqual(config.widths, (24, 48, 96, 192))
        self.assertEqual(config.blocks_per_level, 2)
        self.assertEqual(config.embedding_dim, 16)
        self.assertEqual(config.condition_dim, 128)
        model = build_model(config)
        self.assertEqual(len(model.encoder), 4)
        self.assertTrue(all(len(level) == 2 for level in model.encoder))
        self.assertTrue(all(len(level) == 2 for level in model.decoder))
        self.assertTrue(all(
            isinstance(block, FiLMResidualBlock)
            for level in (*model.encoder, *model.decoder)
            for block in level
        ))

    def test_strict_model_config(self):
        payload = ModelConfig.tiny().to_dict()
        payload["surprise"] = 1
        with self.assertRaisesRegex(Task3Error, "Strict model config"):
            ModelConfig.from_dict(payload)


class ModelTests(unittest.TestCase):
    def test_forward_shape_range_and_near_identity(self):
        model = build_model(ModelConfig.tiny()).eval()
        source = torch.rand(2, 1, 9, 11, 13)
        with torch.inference_mode():
            prediction = model(
                source,
                torch.tensor([0, 2]),
                torch.tensor([0, 3]),
                torch.tensor([4, 1]),
            )
        self.assertEqual(prediction.shape, source.shape)
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertGreaterEqual(float(prediction.min()), 0.0)
        self.assertLessEqual(float(prediction.max()), 1.0)
        torch.testing.assert_close(prediction, source, atol=2e-6, rtol=0.0)

    def test_identity_requires_explicit_pretraining_permission(self):
        model = build_model(ModelConfig.tiny()).eval()
        source = torch.rand(1, 1, 8, 8, 8)
        with self.assertRaisesRegex(Task3Error, "identity"):
            model(source, 0, 2, 2)
        with torch.inference_mode():
            prediction = model(source, 0, 2, 2, allow_identity=True)
        self.assertEqual(prediction.shape, source.shape)

    def test_learned_mask_scalar_replaces_only_masked_voxels(self):
        model = build_model(ModelConfig.tiny())
        with torch.no_grad():
            model.mask_value.fill_(0.25)
        source = torch.ones(1, 1, 2, 2, 2)
        mask = torch.zeros_like(source, dtype=torch.bool)
        mask[..., 0, 0, 0] = True
        corrupted = model.apply_mask(source, mask)
        self.assertEqual(float(corrupted[..., 0, 0, 0].detach()), 0.25)
        self.assertEqual(float(corrupted.sum().detach()), 7.25)


class LossTests(unittest.TestCase):
    def test_block_mask_is_deterministic_and_half_the_blocks(self):
        first = block_mask(
            (2, 1, 24, 24, 24),
            block_size=12,
            fraction=0.5,
            generator=torch.Generator().manual_seed(2026),
        )
        second = block_mask(
            (2, 1, 24, 24, 24),
            block_size=12,
            fraction=0.5,
            generator=torch.Generator().manual_seed(2026),
        )
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(float(first.float().mean()), 0.5)

    def test_masked_loss_normalizes_only_corrupted_foreground(self):
        prediction = torch.tensor([[[[[0.0, 1.0], [0.0, 0.0]]]]])
        target = torch.tensor([[[[[1.0, 0.5], [0.0, 1.0]]]]])
        mask = torch.tensor([[[[[True, True], [True, False]]]]])
        self.assertAlmostEqual(float(masked_reconstruction_loss(prediction, target, mask)), 0.75)

    def test_empty_foreground_is_finite(self):
        prediction = torch.ones(1, 1, 4, 4, 4, requires_grad=True)
        target = torch.zeros_like(prediction)
        empty = torch.zeros_like(prediction, dtype=torch.bool)
        masked = normalized_l1(prediction, target, empty)
        total, terms = fine_tuning_loss(prediction, target, masked=True)
        self.assertEqual(float(masked.detach()), 0.0)
        self.assertEqual(float(terms["foreground_l1"].detach()), 0.0)
        self.assertAlmostEqual(float(total.detach()), 0.1, places=6)
        total.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_foreground_dilation_is_five_cubed(self):
        target = torch.zeros(1, 1, 7, 7, 7)
        target[..., 3, 3, 3] = 1
        foreground = dilated_foreground(target)
        self.assertEqual(int(foreground.sum()), 125)

    def test_tiny_pretrain_and_finetune_updates(self):
        torch.manual_seed(2026)
        model = build_model(ModelConfig.tiny())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        source = torch.rand(1, 1, 8, 8, 8)
        mask = block_mask(
            tuple(source.shape), block_size=4, fraction=0.5,
            generator=torch.Generator().manual_seed(2026),
        )
        prediction = model(model.apply_mask(source, mask), 0, 0, 0, allow_identity=True)
        loss = masked_reconstruction_loss(prediction, source, mask)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        target = torch.rand_like(source)
        prediction = model(source, 0, 0, 1)
        loss, _ = fine_tuning_loss(prediction, target, masked=True)
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
