"""Acceptance tests verifying determinism toggles in constants.py under ON and OFF configurations."""
# pylint: disable=wrong-import-position, import-outside-toplevel, protected-access

import os
import sys

# Must be set before importing torch so MPS fallback enables aten::_cdist_backward.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "tests", "manual"))

import torch

from src.core import constants


def _collect_batches(consume_draws, num_workers, seed=None):
    """Seed RNG, optionally burn draws, and collect batches from a DataLoader."""
    import src.config.config
    import src.learning.augmented_dataset
    import src.learning.pk_sampler
    import trainer_parity as tp
    from src.learning.trainer import new_trainer as nt

    tp.seed_all(seed if seed is not None else constants.RANDOM_SEED)
    base = tp.PseudoIndividualDataset()
    augmentations = nt.setup_augmentations({"augmentations": src.config.config.AUGMENTATIONS})

    # Stands in for backbone / LoRA / head construction consuming RNG draws.
    if consume_draws:
        torch.rand(consume_draws)

    config = tp.build_config("triplet")
    config.training.num_workers = num_workers
    dataset = src.learning.augmented_dataset.AugmentedDataset(
        base, augmentations,
        num_augmentations=config.training.num_augmentations,
        include_original=config.training.include_original,
        rebalance_augmentations=config.training.rebalance_augmentations,
    )
    kwargs = nt._build_dataloader_kwargs(config)
    kwargs["pin_memory"] = False
    sampler = src.learning.pk_sampler.PKSampler(
        base,
        p=config.training.identities_per_batch,
        k=config.training.samples_per_identity,
    )
    loader = torch.utils.data.DataLoader(dataset, sampler=sampler, **kwargs)
    return [images.clone() for images, _ in loader]


def _identical(left, right):
    return len(left) == len(right) and all(
        torch.equal(a, b) for a, b in zip(left, right))


def check_d2_dataloader_generator(num_workers=2):
    """Check that DETERMINISTIC_DATALOADER isolates DataLoader batches from upstream global RNG draws."""
    print(f"D2  DataLoader generator (num_workers={num_workers})")
    print("      toggle | invariant to upstream draws | repeatable")
    results = {}
    original = constants.DETERMINISTIC_DATALOADER
    try:
        for toggle in (False, True):
            constants.DETERMINISTIC_DATALOADER = toggle
            baseline = _collect_batches(consume_draws=0, num_workers=num_workers)
            shifted = _collect_batches(consume_draws=7, num_workers=num_workers)
            repeat = _collect_batches(consume_draws=0, num_workers=num_workers)
            invariant = _identical(baseline, shifted)
            repeatable = _identical(baseline, repeat)
            results[toggle] = (invariant, repeatable)
            print(f"      {'ON ' if toggle else 'OFF'}    | "
                  f"{str(invariant):27s} | {repeatable}")
    finally:
        constants.DETERMINISTIC_DATALOADER = original

    ok = (results[False][0] is False and results[True][0] is True
          and results[False][1] and results[True][1])
    print(f"      -> {'PASS' if ok else 'FAIL'}: OFF must be fragile and ON must be invariant\n")
    return ok


def check_d1_strict_determinism():
    """Verify that determinism flags pin algorithm selection and reject nondeterministic ops."""
    import io
    import contextlib
    import warnings as warnings_mod

    from src.core import utils

    print("D1  strict determinism flags")
    original = constants.STRICT_DETERMINISM
    try:
        constants.STRICT_DETERMINISM = False
        torch.use_deterministic_algorithms(False)
        utils.seed_everything()
        off_state = (torch.are_deterministic_algorithms_enabled(),
                     torch.backends.cudnn.deterministic)
        print(f"      OFF -> deterministic_algorithms={off_state[0]}, "
              f"cudnn.deterministic={off_state[1]}")

        constants.STRICT_DETERMINISM = True
        with contextlib.redirect_stdout(io.StringIO()):
            utils.seed_everything()
        on_state = (torch.are_deterministic_algorithms_enabled(),
                    torch.backends.cudnn.deterministic)
        hashseed = os.environ.get("PYTHONHASHSEED")
        print(f"      ON  -> deterministic_algorithms={on_state[0]}, "
              f"cudnn.deterministic={on_state[1]}, PYTHONHASHSEED={hashseed}")

        flags_ok = (off_state == (False, False) and on_state == (True, True)
                    and hashseed == str(constants.RANDOM_SEED))

        # Scan a real training run for ops with no deterministic kernel. Any
        # warning here names a genuine reproducibility hole worth knowing about.
        import tempfile
        import trainer_parity as tp
        with warnings_mod.catch_warnings(record=True) as caught:
            warnings_mod.simplefilter("always")
            with contextlib.redirect_stdout(io.StringIO()), \
                    tempfile.TemporaryDirectory() as workdir:
                tp.run_scenario("triplet", torch.device("cpu"), workdir, loss="triplet")
        nondet = [str(w.message) for w in caught
                  if "does not have a deterministic implementation" in str(w.message)]
        print(f"      nondeterministic ops hit during a run: {len(nondet)}")
        for message in nondet[:3]:
            print(f"        - {message.splitlines()[0][:90]}")
    finally:
        constants.STRICT_DETERMINISM = original
        torch.use_deterministic_algorithms(False)

    print(f"      -> {'PASS' if flags_ok else 'FAIL'}: flags must be off when OFF "
          f"and on when ON\n")
    return flags_ok


def _train_and_capture_batches(extra_construction_draws):
    """Run a short training loop after burning RNG draws and return the batches seen."""
    import io
    import contextlib
    import tempfile

    import trainer_parity as tp
    from src.learning.trainer import new_trainer as nt

    tp.seed_all()
    config, model, strategy, loader, dataset = tp.build_scenario(
        "triplet", torch.device("cpu"))

    # Simulate extra RNG draws during model construction.
    if extra_construction_draws:
        torch.rand(extra_construction_draws)

    seen = []
    original_process = strategy.process_batch

    def recording(model_arg, batch, device_arg):
        seen.append(batch[0].clone())
        return original_process(model_arg, batch, device_arg)

    strategy.process_batch = recording
    trainer = nt.RobustTrainer(model, strategy, torch.device("cpu"), config,
                               dataset, dataset)
    with contextlib.redirect_stdout(io.StringIO()), tempfile.TemporaryDirectory() as workdir:
        trainer.train(loader, workdir)
    return seen


def check_d3_reseed_before_train():
    """Verify that reseeding isolates data-loader stream reproducibility from model construction RNG draws."""
    print("D3  re-seed before train()")
    print("      toggle | data stream survives a construction change | repeatable")
    results = {}
    original = constants.RESEED_BEFORE_TRAIN
    try:
        for toggle in (False, True):
            constants.RESEED_BEFORE_TRAIN = toggle
            baseline = _train_and_capture_batches(0)
            altered = _train_and_capture_batches(5)
            repeat = _train_and_capture_batches(0)
            invariant = _identical(baseline, altered)
            repeatable = _identical(baseline, repeat)
            results[toggle] = (invariant, repeatable)
            print(f"      {'ON ' if toggle else 'OFF'}    | "
                  f"{str(invariant):42s} | {repeatable}")
    finally:
        constants.RESEED_BEFORE_TRAIN = original

    ok = (results[False][0] is False and results[True][0] is True
          and results[False][1] and results[True][1])
    print(f"      -> {'PASS' if ok else 'FAIL'}: OFF must be fragile and ON must be invariant\n")
    return ok


def _train_on_device(device, disable_amp):
    """Train one short scenario on `device` and return the probe logits."""
    import io
    import contextlib
    import tempfile

    import trainer_parity as tp
    from src.learning.trainer import new_trainer as nt

    tp.seed_all()
    if device.type == "mps":
        torch.mps.manual_seed(constants.RANDOM_SEED)

    config, model, strategy, loader, dataset = tp.build_scenario("triplet", device)
    trainer = nt.RobustTrainer(model, strategy, device, config, dataset, dataset)
    if disable_amp:
        trainer.use_amp = False

    with contextlib.redirect_stdout(io.StringIO()), tempfile.TemporaryDirectory() as workdir:
        trainer.train(loader, workdir)

    probe = torch.rand(4, 3, tp.IMAGE_SIZE, tp.IMAGE_SIZE,
                       generator=torch.Generator().manual_seed(
                           constants.RANDOM_SEED + 1)).to(device)
    model.eval()
    with torch.no_grad():
        return model(probe).float().cpu()


def check_d4_mps_reproducibility():
    """Verify MPS training reproducibility by comparing bitwise parity across repeated runs with AMP on and off."""
    print("D4  MPS reproducibility")
    if not torch.backends.mps.is_available():
        print("      SKIP: no MPS on this machine\n")
        return True

    device = torch.device("mps")
    results = {}
    for disable_amp in (False, True):
        label = "AMP off" if disable_amp else "AMP on (production)"
        first = _train_on_device(device, disable_amp)
        second = _train_on_device(device, disable_amp)
        identical = torch.equal(first, second)
        results[label] = identical
        print(f"      MPS vs MPS, {label:20s} -> identical: {identical}")

    cpu = _train_on_device(torch.device("cpu"), True)
    mps = _train_on_device(device, True)
    cross = torch.equal(cpu, mps)
    delta = (cpu - mps).abs().max().item()
    print(f"      MPS vs CPU, AMP off              -> identical: {cross} "
          f"(max abs {delta:.2e}, expected to differ)")

    ok = all(results.values())
    print(f"      -> {'PASS' if ok else 'FAIL'}: MPS must reproduce itself run to run\n")
    return ok


def main():
    """Run every determinism acceptance check."""
    torch.set_num_threads(1)
    checks = [check_d2_dataloader_generator, check_d1_strict_determinism,
              check_d3_reseed_before_train, check_d4_mps_reproducibility]
    results = [check() for check in checks]
    passed = all(results)
    print("=" * 60)
    print(f"{sum(results)}/{len(results)} determinism checks passed")
    print("=" * 60)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
