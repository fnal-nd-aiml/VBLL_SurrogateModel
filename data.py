# minerva_vbll/data.py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

PARTICLE_NAMES = ['muon', 'proton']
PARTICLE_TYPE  = {'muon': 0, 'proton': 1}
COMPONENTS     = ['E', 'px', 'py', 'pz']


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_minerva_csv(path: str) -> pd.DataFrame:
    """
    Parse the v0 two-row event format described in `note`.

    Assumption: an event header is the only row with exactly two comma-separated
    fields, followed by one truth row and one reco row with fixed column offsets.
    This parser intentionally stays strict about those offsets because the
    August dataset format is expected to add PID and matching fields.
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
            truth, reco = lines[i].split(','), lines[i+1].split(',')
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


# ── Cleaning ──────────────────────────────────────────────────────────────────

def split_outliers(df: pd.DataFrame, muon_reco_E_max: float = 20_000):
    bad = df['reco_muon_E'] > muon_reco_E_max
    print(f"Flagged {bad.sum()} outlier events ({100*bad.mean():.1f}%)")
    return df[~bad].reset_index(drop=True), df[bad].reset_index(drop=True)


# ── Normalisation ─────────────────────────────────────────────────────────────

class Normaliser:
    """
    Separate statistics for inputs (truth) and outputs (reco).

    This is critical for preventing energy scale collapse:
    truth_E and reco_E have different distributions (reco has heavier tails,
    catastrophic failures, dEdx biases). Normalising both by truth_E std
    leaves reco_E still dominant in the VBLL output space.

    By normalising reco targets by their own mean/std, every output component
    is brought to roughly unit-variance in the space the VBLL head sees,
    giving the VBLL noise prior a fair starting point across all four
    4-momentum components.

    input_stats  : used to normalise truth 4-vectors (model input)
    output_stats : used to normalise reco  4-vectors (VBLL regression target)
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

    # ── Input (truth) transforms ──────────────────────────────────────────────

    def transform_input(self, particle: str, comp: str, v):
        mu, sig = self.input_stats[(particle, comp)]
        return (v - mu) / (sig + 1e-8)

    def inverse_input(self, particle: str, comp: str, v):
        mu, sig = self.input_stats[(particle, comp)]
        return v * (sig + 1e-8) + mu

    # ── Output (reco) transforms ──────────────────────────────────────────────

    def transform_output(self, particle: str, comp: str, v):
        mu, sig = self.output_stats[(particle, comp)]
        return (v - mu) / (sig + 1e-8)

    def inverse_output(self, particle: str, comp: str, v):
        mu, sig = self.output_stats[(particle, comp)]
        return v * (sig + 1e-8) + mu

    def output_std(self, particle: str, comp: str) -> float:
        """Raw reco std in MeV — used to convert normalised σ back to physical units."""
        return self.output_stats[(particle, comp)][1]


# ── Dataset ───────────────────────────────────────────────────────────────────

class MINERvADataset(Dataset):
    """
    Each event → 2 samples (muon + proton).

    Inputs  (truth 4-vec) : normalised by input_stats  (truth distribution)
    Targets (reco  4-vec) : normalised by output_stats (reco  distribution)

    This ensures the VBLL head sees all four components on a comparable
    scale, preventing any single component from dominating the noise parameter.

    v1 extension points:
        - Add pid field  : FloatTensor (6,) — likelihood scores per prong
        - Add matched    : BoolTensor   — False when truth has no matched prong
        - target_idx will differ from type_idx for misidentified particles
    """
    def __init__(self, df: pd.DataFrame, normaliser: Normaliser):
        self.samples = []
        for _, row in df.iterrows():
            for particle, pidx in PARTICLE_TYPE.items():
                truth = np.array([row[f'truth_{particle}_{c}'] for c in COMPONENTS],
                                 dtype=np.float32)
                reco  = np.array([row[f'reco_{particle}_{c}']  for c in COMPONENTS],
                                 dtype=np.float32)

                for j, comp in enumerate(COMPONENTS):
                    truth[j] = normaliser.transform_input(particle, comp, truth[j])
                    reco[j]  = normaliser.transform_output(particle, comp, reco[j])

                self.samples.append((
                    torch.tensor(pidx,  dtype=torch.long),
                    torch.tensor(truth, dtype=torch.float32),
                    torch.tensor(pidx,  dtype=torch.long),    # target_idx = type_idx in v0
                    torch.tensor(reco,  dtype=torch.float32),
                ))

    def __len__(self):         return len(self.samples)
    def __getitem__(self, i):  return self.samples[i]


# ── Public factory ────────────────────────────────────────────────────────────

def build_loaders(path: str, batch_size: int = 64, outlier_threshold: float = 20_000):
    df_raw           = parse_minerva_csv(path)
    df_clean, df_out = split_outliers(df_raw, outlier_threshold)

    n        = len(df_clean)
    # Keep this deterministic while the source CSV has no explicit split field.
    # If the CSV is ordered by run, event quality, or production campaign, replace
    # this with a seeded random split or a domain-provided train/validation flag.
    train_df = df_clean.iloc[:n//2].reset_index(drop=True)
    val_df   = df_clean.iloc[n//2:].reset_index(drop=True)

    normaliser = Normaliser(train_df)

    train_loader   = DataLoader(MINERvADataset(train_df, normaliser),
                                batch_size=batch_size, shuffle=True)
    val_loader     = DataLoader(MINERvADataset(val_df,   normaliser),
                                batch_size=batch_size, shuffle=False)
    outlier_loader = DataLoader(MINERvADataset(df_out,   normaliser),
                                batch_size=batch_size, shuffle=False)

    # n_train_per_particle: each event contributes one sample per particle type
    n_train_per_particle = len(train_df)

    return train_loader, val_loader, outlier_loader, normaliser, n_train_per_particle
