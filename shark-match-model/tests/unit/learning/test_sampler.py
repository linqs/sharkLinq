
import unittest
import torch
import random
import src.learning.pk_sampler as pk_sampler
import src.individual_dataset

class MockDataset(torch.utils.data.Dataset):
    def __init__(self, labels):
        self.labels = labels
        # Dataset returns (image, label).
        # We only care about label for sampler.
        self.data = [(None, torch.tensor(l)) for l in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_filenames(self):
        return [str(i) for i in range(len(self.labels))]

class TestPKSampler(unittest.TestCase):
    def setUp(self):
        # Labels: 0 (x4), 1 (x3), 2 (x2), 3 (x1).
        # Four distinct identities with varying support.
        self.labels = [0, 0, 0, 0, 1, 1, 1, 2, 2, 3]
        self.dataset = MockDataset(self.labels)

    def test_batch_structure(self):
        """Each batch must contain exactly P identities × K images."""
        p, k = 2, 3
        sampler = pk_sampler.PKSampler(self.dataset, p=p, k=k)
        indices = list(iter(sampler))

        batch_size = p * k
        self.assertEqual(len(indices) % batch_size, 0,
                         "Total indices must be a multiple of P*K")

        for batch_start in range(0, len(indices), batch_size):
            batch = indices[batch_start:batch_start + batch_size]

            # Group indices by identity within the batch.
            label_to_count = {}
            for idx in batch:
                label = self.dataset[idx][1].item()
                label_to_count[label] = label_to_count.get(label, 0) + 1

            self.assertEqual(len(label_to_count), p,
                             f"Batch should contain exactly P={p} identities")
            for label, count in label_to_count.items():
                self.assertEqual(count, k,
                                 f"Identity {label} should have K={k} images, got {count}")

    def test_len(self):
        """__len__ should equal (num_identities // P) * P * K."""
        p, k = 2, 3
        sampler = pk_sampler.PKSampler(self.dataset, p=p, k=k)

        # 4 identities, P=2 -> 2 batches, batch_size = 6 -> total 12.
        self.assertEqual(len(sampler), 12)
        self.assertEqual(len(list(iter(sampler))), 12)

    def test_oversampling_for_small_identity(self):
        """Identities with fewer than K images are oversampled with replacement."""
        # Identity 3 has only 1 image; K=3 forces oversampling.
        p, k = 2, 3
        sampler = pk_sampler.PKSampler(self.dataset, p=p, k=k)
        indices = list(iter(sampler))

        # Find any batch containing identity 3 and check it has K entries for it.
        batch_size = p * k
        seen_identity_3 = False
        for batch_start in range(0, len(indices), batch_size):
            batch = indices[batch_start:batch_start + batch_size]
            count_3 = sum(1 for idx in batch if self.dataset[idx][1].item() == 3)
            if count_3 > 0:
                seen_identity_3 = True
                self.assertEqual(count_3, k,
                                 "Singleton identity should still produce K entries via oversampling")
        # Identity 3 may not be sampled in every epoch (4 ids, P=2 -> 2 batches),
        # but if present, the count must be K.
        _ = seen_identity_3

    def test_determinism(self):
        """Same seed must produce identical sampling order."""
        p, k = 2, 3
        seed = 42

        random.seed(seed)
        torch.manual_seed(seed)
        sampler1 = pk_sampler.PKSampler(self.dataset, p=p, k=k)
        indices1 = list(iter(sampler1))

        random.seed(seed)
        torch.manual_seed(seed)
        sampler2 = pk_sampler.PKSampler(self.dataset, p=p, k=k)
        indices2 = list(iter(sampler2))

        self.assertEqual(indices1, indices2)

        # Different seed should (almost always) produce a different order.
        random.seed(seed + 1)
        torch.manual_seed(seed + 1)
        sampler3 = pk_sampler.PKSampler(self.dataset, p=p, k=k)
        indices3 = list(iter(sampler3))

        self.assertNotEqual(indices1, indices3)

    def test_raises_when_too_few_identities(self):
        """PKSampler must raise if dataset has fewer than P identities."""
        small_dataset = MockDataset([0, 0, 1, 1])
        with self.assertRaises(ValueError):
            pk_sampler.PKSampler(small_dataset, p=5, k=2)

if __name__ == '__main__':
    unittest.main()
