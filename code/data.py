# minerva_vbll/data.py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import re

PARTICLE_NAMES = ['muon', 'proton']
PARTICLE_TYPE  = {'muon': 0, 'proton': 1}
COMPONENTS     = ['E', 'px', 'py', 'pz']


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_minerva_csv(path: str) -> pd.DataFrame:
    """
    Parse one MINERvA surrogate-model CSV.

    Format:
      event header row (exactly 2 comma-separated fields)
      truth row
      reco row

    The truth/reco rows use the fixed offsets expected by the current v0
    dataset format.
    """
    records = []
    with open(path) as f:
        lines = [l.strip() for l in f]

    i = 0
    while i < len(lines):
        parts = lines[i].split(',')

        if len(parts) == 2:
            event_id = parts[0]
            i += 1

            if i + 1 >= len(lines):
                break

            truth = lines[i].split(',')
            reco  = lines[i + 1].split(',')
            i += 2

            records.append({
                'event_id': event_id,
                **{f'truth_muon_{c}':   float(truth[2+j]) for j, c in enumerate(COMPONENTS)},
                **{f'truth_proton_{c}': float(truth[8+j]) for j, c in enumerate(COMPONENTS)},
                **{f'reco_muon_{c}':    float(reco[2+j])  for j, c in enumerate(COMPONENTS)},
                **{f'reco_proton_{c}':  float(reco[8+j])  for j, c in enumerate(COMPONENTS)},
            })
        else:
            i += 1

    return pd.DataFrame(records)


def parse_minerva_input(path: str) -> pd.DataFrame:
    """
    Accept either:

      1. a single MINERvA CSV file, or
      2. a directory containing chunked files named
         masteranadev_selected_events_<number>.csv

    Each chunk is parsed independently with parse_minerva_csv(), then the
    already-parsed event DataFrames are concatenated.

    Files such as masteranadev_selected_events_x60.csv are intentionally
    ignored so an accidentally-created combined file is not double counted.
    """
    input_path = Path(path)

    if input_path.is_file():
        print(f"Loading single input file: {input_path}")
        df = parse_minerva_csv(str(input_path))
        print(f"Parsed {len(df):,} events")
        return df

    if input_path.is_dir():
        file_re = re.compile(r"^masteranadev_selected_events_\d+\.csv$")

        files = sorted(
            f for f in input_path.iterdir()
            if f.is_file() and file_re.match(f.name)
        )

        if not files:
            raise FileNotFoundError(
                f"No files matching masteranadev_selected_events_<number>.csv "
                f"were found in {input_path}"
            )

        print(f"Found {len(files)} input files in {input_path}")

        frames = []
        total = 0

        for i, f in enumerate(files, start=1):
            df = parse_minerva_csv(str(f))
            frames.append(df)
            total += len(df)
            print(
                f"  [{i:02d}/{len(files):02d}] {f.name}: "
                f"{len(df):,} events  (running total {total:,})"
            )

        df_all = pd.concat(frames, ignore_index=True)
        print(f"Combined parsed dataset: {len(df_all):,} events")
        return df_all

    raise FileNotFoundError(
        f"Input path does not exist or is not a file/directory: {input_path}"
    )


# ── Cleaning ──────────────────────────────────────────────────────────────────

def split_outliers(df: pd.DataFrame, muon_reco_E_max: float = 20_000):
    bad = df['reco_muon_E'] > muon_reco_E_max
    print(f"Flagged {bad.sum()} outlier events ({100*bad.mean():.1f}%)")
    return df[~bad].reset_index(drop=True), df[bad].reset_index(drop=True)


# ── Normalisation ─────────────────────────────────────────────────────────────

class Normaliser:
    """
    Separate statistics for inputs (truth) and outputs (reco).

    input_stats  : used to normalise truth 4-vectors
    output_stats : used to normalise reco  4-vectors
    """
    def __init__(self, train_df: pd.DataFrame):
        self.input_stats  = {}
        self.output_stats = {}

        for particle in PARTICLE_NAMES:
            for comp in COMPONENTS:
                t_col = f'truth_{particle}_{comp}'
                r_col = f'reco_{particle}_{comp}'

                self.input_stats[(particle, comp)] = (
                    float(train_df[t_col].mean()),
                    float(train_df[t_col].std()),
                )
                self.output_stats[(particle, comp)] = (
                    float(train_df[r_col].mean()),
                    float(train_df[r_col].std()),
                )

    def transform_input(self, particle: str, comp: str, v):
        mu, sig = self.input_stats[(particle, comp)]
        return (v - mu) / (sig + 1e-8)

    def inverse_input(self, particle: str, comp: str, v):
        mu, sig = self.input_stats[(particle, comp)]
        return v * (sig + 1e-8) + mu

    def transform_output(self, particle: str, comp: str, v):
        mu, sig = self.output_stats[(particle, comp)]
        return (v - mu) / (sig + 1e-8)

    def inverse_output(self, particle: str, comp: str, v):
        mu, sig = self.output_stats[(particle, comp)]
        return v * (sig + 1e-8) + mu

    def output_std(self, particle: str, comp: str) -> float:
        return self.output_stats[(particle, comp)][1]


# ── Dataset ───────────────────────────────────────────────────────────────────

class MINERvADataset(Dataset):
    """
    Each event -> 2 samples (muon + proton).

    Inputs  : truth 4-vector, normalized with truth statistics.
    Targets : reco 4-vector, normalized with reco statistics.
    """
    def __init__(self, df: pd.DataFrame, normaliser: Normaliser):
        self.samples = []

        for _, row in df.iterrows():
            for particle, pidx in PARTICLE_TYPE.items():
                truth = np.array(
                    [row[f'truth_{particle}_{c}'] for c in COMPONENTS],
                    dtype=np.float32
                )
                reco = np.array(
                    [row[f'reco_{particle}_{c}'] for c in COMPONENTS],
                    dtype=np.float32
                )

                for j, comp in enumerate(COMPONENTS):
                    truth[j] = normaliser.transform_input(
                        particle, comp, truth[j]
                    )
                    reco[j] = normaliser.transform_output(
                        particle, comp, reco[j]
                    )

                self.samples.append((
                    torch.tensor(pidx, dtype=torch.long),
                    torch.tensor(truth, dtype=torch.float32),
                    torch.tensor(pidx, dtype=torch.long),
                    torch.tensor(reco, dtype=torch.float32),
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


# ── Public factory ────────────────────────────────────────────────────────────

def build_loaders(
    path: str,
    batch_size: int = 64,
    outlier_threshold: float = 20_000,
    train_fraction: float = 0.50,
    split_seed: int = 42,
):
    """
    Build train, validation, and outlier loaders.

    IMPORTANT:
    Clean events are shuffled BEFORE splitting. This avoids the old behavior
    where concatenated files 00,01,... were effectively split by production
    order (early files -> train, later files -> validation).

    The split is reproducible because the shuffle uses split_seed.

    Defaults keep the old 50/50 train/validation fraction so dataset size is
    the only intended change relative to the original study.
    """

    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be between 0 and 1.")

    df_raw = parse_minerva_input(path)
    df_clean, df_out = split_outliers(df_raw, outlier_threshold)

    # NEW: reproducibly randomise clean events before splitting.
    df_clean = df_clean.sample(
        frac=1.0,
        random_state=split_seed
    ).reset_index(drop=True)

    n_clean = len(df_clean)
    n_train = int(train_fraction * n_clean)

    train_df = df_clean.iloc[:n_train].copy().reset_index(drop=True)
    val_df   = df_clean.iloc[n_train:].copy().reset_index(drop=True)

    print(
        f"Random train/val split: "
        f"{len(train_df):,} train / {len(val_df):,} val clean events "
        f"(train_fraction={train_fraction:.2f}, seed={split_seed})"
    )

    # Fit ALL normalization constants using TRAINING DATA ONLY.
    normaliser = Normaliser(train_df)

    # Seed the DataLoader shuffle too, so the run is easier to reproduce.
    loader_generator = torch.Generator()
    loader_generator.manual_seed(split_seed)

    train_loader = DataLoader(
        MINERvADataset(train_df, normaliser),
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
    )

    val_loader = DataLoader(
        MINERvADataset(val_df, normaliser),
        batch_size=batch_size,
        shuffle=False,
    )

    outlier_loader = DataLoader(
        MINERvADataset(df_out, normaliser),
        batch_size=batch_size,
        shuffle=False,
    )

    # Every clean event contributes exactly one muon and one proton sample.
    n_train_per_particle = len(train_df)

    return (
        train_loader,
        val_loader,
        outlier_loader,
        normaliser,
        n_train_per_particle,
    )
