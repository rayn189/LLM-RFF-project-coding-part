import csv
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from pt_dataset import PTWindowDataset
from model import TeacherRFFLLM, StudentLiteProxy, DistillationLoss
from ppo_scheduler import PPOTemperatureScheduler, PPOTempConfig
from metrics_export import measure_deployment_metrics, count_params_m


PT_ROOT = r"C:\Users\RAYNES\.cache\modelscope\datasets\zzcyber--DRFF-R1\snapshots\master\pt_windows"
WINDOW_SIZE = 4096
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-4
WEIGHT_DECAY = 1e-4
TEMPERATURE = 4.0
ALPHA = 0.5
NUM_WORKERS = 0
SEED = 42
USE_PPO = True
RUN_BASELINE = True
METRICS_DIR = "metrics_runs"
PLOTS_DIR = os.path.join(METRICS_DIR, "plots")
EXCEL_PATH = os.path.join(METRICS_DIR, "deployment_summary.xlsx")


def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


DEVICE = get_device()


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total = 0
    correct = 0
    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


def train_teacher_epoch(model, loader, optimizer):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def train_student_epoch(teacher, student, loader, optimizer, kd_criterion):
    teacher.eval()
    student.train()
    total_loss = total_ce = total_kd = 0.0
    total_correct = 0
    total_samples = 0
    kl_values = []
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        with torch.no_grad():
            teacher_logits = teacher(x)
        student_logits = student(x)
        loss, ce_loss, kd_loss, batch_kl = kd_criterion(student_logits, teacher_logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_ce += ce_loss.item()
        total_kd += kd_loss.item()
        total_correct += (student_logits.argmax(dim=1) == y).sum().item()
        total_samples += y.size(0)
        kl_values.append(float(batch_kl.item()))
    steps = max(len(loader), 1)
    batch_acc = total_correct / max(total_samples, 1)
    kl_mean = float(np.mean(kl_values)) if kl_values else 0.0
    kl_std = float(np.std(kl_values)) if kl_values else 0.0
    return total_loss / steps, total_ce / steps, total_kd / steps, batch_acc, kl_mean, kl_std


def prepare_loaders(dataset):
    n_total = len(dataset)
    n_train = int(0.6 * n_total)
    n_val = int(0.2 * n_total)
    n_test = n_total - n_train - n_val
    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(SEED)
    )
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=(DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=(DEVICE.type == "cuda"),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=(DEVICE.type == "cuda"),
    )
    return train_loader, val_loader, test_loader


def write_excel_summary(rows_by_tag, deployments_by_tag):
    try:
        import pandas as pd
    except ImportError:
        print("pandas not installed; skipping Excel export")
        return

    os.makedirs(METRICS_DIR, exist_ok=True)
    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as writer:
        summary_rows = []
        for tag, rows in rows_by_tag.items():
            df = pd.DataFrame(rows)
            df.to_excel(writer, sheet_name=f"{tag}_epochs"[:31], index=False)
            dep = deployments_by_tag[tag]
            summary_rows.append({
                "run": tag,
                "accuracy": dep.accuracy,
                "delay_ms_per_sample": dep.delay_ms_per_sample,
                "params_m": dep.params_m,
                "complexity_score": dep.complexity_score,
                "peak_vram_mb": dep.peak_vram_mb,
                "benchmark_batch_size": dep.benchmark_batch_size,
                "benchmark_samples": dep.benchmark_samples,
                "warmup_batches": dep.warmup_batches,
                "measurement_batches": dep.measurement_batches,
            })
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="summary", index=False)
    print(f"Saved Excel summary to {EXCEL_PATH}")


def run_experiment(dataset, num_classes, use_ppo: bool, tag: str):
    train_loader, val_loader, test_loader = prepare_loaders(dataset)
    x0, _ = dataset[0]
    inferred_window = x0.shape[-1]

    teacher = TeacherRFFLLM(inferred_window, num_classes).to(DEVICE)
    student = StudentLiteProxy(num_classes).to(DEVICE)
    print(f"[{tag}] Teacher params (M):", count_params_m(teacher))
    print(f"[{tag}] Student params (M):", count_params_m(student))

    teacher_ckpt = "teacher_best.pt"
    best_teacher_val = 0.0
    if os.path.exists(teacher_ckpt):
        teacher.load_state_dict(torch.load(teacher_ckpt, map_location=DEVICE))
        print(f"[{tag}] Loaded teacher checkpoint: {teacher_ckpt}")
    else:
        print(f"[{tag}] Training teacher from scratch...")
        teacher_opt = torch.optim.AdamW(teacher.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        for epoch in range(1, EPOCHS + 1):
            loss = train_teacher_epoch(teacher, train_loader, teacher_opt)
            val_acc = evaluate(teacher, val_loader)
            print(f"[{tag}][Teacher] Epoch {epoch:03d} | loss={loss:.4f} val_acc={val_acc:.4f}")
            if val_acc > best_teacher_val:
                best_teacher_val = val_acc
                torch.save(teacher.state_dict(), teacher_ckpt)
                print(f"[{tag}] Saved teacher checkpoint to {teacher_ckpt}")
        teacher.load_state_dict(torch.load(teacher_ckpt, map_location=DEVICE))
        print(f"[{tag}] Best teacher val acc: {best_teacher_val:.4f}")

    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    student_opt = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    kd_criterion = DistillationLoss(temperature=TEMPERATURE, alpha=ALPHA)
    ppo = PPOTemperatureScheduler(PPOTempConfig()).to(DEVICE) if use_ppo else None

    best_student_val = 0.0
    student_ckpt = f"student_kd_best_{tag}.pt"
    rows = []
    for epoch in range(1, EPOCHS + 1):
        temp = TEMPERATURE
        ppo_loss = 0.0
        reward = 0.0
        if ppo is not None:
            state = [
                best_teacher_val,
                best_student_val,
                epoch / max(EPOCHS, 1),
                0.0,
                0.0,
                0.0,
                0.0,
            ]
            temp, _ = ppo.sample_temperature(state, device=DEVICE)
            kd_criterion.temperature = temp
            print(f"[{tag}][PPO] epoch={epoch:03d} temperature={temp:.3f}")

        train_loss, train_ce, train_kd, batch_acc, kl_mean, kl_std = train_student_epoch(
            teacher, student, train_loader, student_opt, kd_criterion
        )
        val_acc = evaluate(student, val_loader)
        previous_acc = rows[-1]["val_acc"] if rows else 0.0
        previous_kl = rows[-1]["kl_mean"] if rows else kl_mean
        acc_delta = val_acc - previous_acc
        kl_delta = kl_mean - previous_kl

        reward = (batch_acc - 0.5) + np.log1p(10.0 * (kl_mean - 0.05) ** 2) + 0.01 * abs(temp - (rows[-1]["temperature"] if rows else TEMPERATURE))
        if ppo is not None:
            ppo.store_transition(reward)
            ppo_loss = ppo.update()
            print(f"[{tag}][PPO] reward={reward:.4f} ppo_loss={ppo_loss:.4f}")

        epoch_deployment = measure_deployment_metrics(student, val_loader, DEVICE, val_acc)
        print(
            f"[{tag}][Deployment] Epoch {epoch:03d} | "
            f"accuracy={epoch_deployment.accuracy:.4f} "
            f"delay_ms_per_sample={epoch_deployment.delay_ms_per_sample:.4f} "
            f"params_m={epoch_deployment.params_m:.6f} "
            f"complexity={epoch_deployment.complexity_score:.6f}"
        )
        print(f"[{tag}][Student KD] Epoch {epoch:03d} | loss={train_loss:.4f} ce={train_ce:.4f} kd={train_kd:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_student_val:
            best_student_val = val_acc
            torch.save(student.state_dict(), student_ckpt)
            print(f"[{tag}] Saved student checkpoint to {student_ckpt}")

        rows.append({
            "epoch": epoch,
            "temperature": temp,
            "train_loss": train_loss,
            "train_ce": train_ce,
            "train_kd": train_kd,
            "batch_acc": batch_acc,
            "val_acc": val_acc,
            "kl_mean": kl_mean,
            "kl_std": kl_std,
            "acc_delta": acc_delta,
            "kl_delta": kl_delta,
            "reward": reward,
            "ppo_loss": ppo_loss,
            "deployment_accuracy": epoch_deployment.accuracy,
            "delay_ms_per_sample": epoch_deployment.delay_ms_per_sample,
            "params_m": epoch_deployment.params_m,
            "complexity_score": epoch_deployment.complexity_score,
            "peak_vram_mb": epoch_deployment.peak_vram_mb,
            "benchmark_batch_size": epoch_deployment.benchmark_batch_size,
            "benchmark_samples": epoch_deployment.benchmark_samples,
            "warmup_batches": epoch_deployment.warmup_batches,
            "measurement_batches": epoch_deployment.measurement_batches,
        })

    if os.path.exists(student_ckpt):
        student.load_state_dict(torch.load(student_ckpt, map_location=DEVICE))
    test_acc = evaluate(student, test_loader)
    print(f"[{tag}] Best student val acc: {best_student_val:.4f}")
    print(f"[{tag}] Student test acc: {test_acc:.4f}")

    deployment = measure_deployment_metrics(student, test_loader, DEVICE, test_acc)
    print(
        f"[{tag}] Final deployment | accuracy={deployment.accuracy:.4f} "
        f"delay_ms_per_sample={deployment.delay_ms_per_sample:.4f} "
        f"params_m={deployment.params_m:.6f} "
        f"complexity={deployment.complexity_score:.6f}"
    )

    os.makedirs(METRICS_DIR, exist_ok=True)
    csv_path = os.path.join(METRICS_DIR, f"metrics_{tag}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + ["test_acc"])
        writer.writeheader()
        for r in rows:
            writer.writerow({**r, "test_acc": test_acc})

    return rows, test_acc, deployment


def main():
    set_seed(SEED)
    dataset = PTWindowDataset(PT_ROOT)
    num_classes = len(set(sample["label"] for sample in dataset.samples))
    print("Num classes:", num_classes)
    print("Total windows:", len(dataset))

    rows_by_tag = {}
    deployments_by_tag = {}

    if RUN_BASELINE:
        print("\n=== Running no-PPO baseline ===")
        baseline_rows, baseline_test, baseline_dep = run_experiment(dataset, num_classes, use_ppo=False, tag="baseline")
        rows_by_tag["baseline"] = baseline_rows
        deployments_by_tag["baseline"] = baseline_dep

    if USE_PPO:
        print("\n=== Running PPO version ===")
        ppo_rows, ppo_test, ppo_dep = run_experiment(dataset, num_classes, use_ppo=True, tag="ppo")
        rows_by_tag["ppo"] = ppo_rows
        deployments_by_tag["ppo"] = ppo_dep

    write_excel_summary(rows_by_tag, deployments_by_tag)
    print("Done.")


if __name__ == "__main__":
    main()
