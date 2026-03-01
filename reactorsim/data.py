from dataclasses import dataclass
from typing import List
import pandas as pd


@dataclass(frozen=True)
class StirredReactorCFDRow:
    volume: float
    reactor_diameter: float
    liquid_height: float
    bottom_depth: float
    n_impeller: int
    diameter_impeller_1: float
    ax_pos_impeller_1: float
    n_blades_1: int
    pitch_angle_1: float
    diameter_impeller_2: float
    ax_pos_impeller_2: float
    n_blades_2: int
    pitch_angle_2: float
    diameter_impeller_3: float
    ax_pos_impeller_3: float
    n_blades_3: int
    pitch_angle_3: float
    n_diptubes: int
    n_baffles: int
    rpm: float
    strainRate_p50: float
    epsilon_p50: float
    k_p50: float
    Umag_p50: float


def load_stirred_reactor_cfd_csv(path: str) -> List[StirredReactorCFDRow]:
    df = pd.read_csv(path)
    # Ensure expected columns exist
    expected = [
        "volume","reactor_diameter","liquid_height","bottom_depth","n_impeller",
        "diameter_impeller_1","ax_pos_impeller_1","n_blades_1","pitch_angle_1",
        "diameter_impeller_2","ax_pos_impeller_2","n_blades_2","pitch_angle_2",
        "diameter_impeller_3","ax_pos_impeller_3","n_blades_3","pitch_angle_3",
        "n_diptubes","n_baffles","rpm","strainRate_p50","epsilon_p50","k_p50","Umag_p50",
    ]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    rows = []
    for _, row in df.iterrows():
        rows.append(StirredReactorCFDRow(
            volume=row["volume"],
            reactor_diameter=row["reactor_diameter"],
            liquid_height=row["liquid_height"],
            bottom_depth=row["bottom_depth"],
            n_impeller=int(row["n_impeller"]),
            diameter_impeller_1=row["diameter_impeller_1"],
            ax_pos_impeller_1=row["ax_pos_impeller_1"],
            n_blades_1=int(row["n_blades_1"]),
            pitch_angle_1=row["pitch_angle_1"],
            diameter_impeller_2=row["diameter_impeller_2"],
            ax_pos_impeller_2=row["ax_pos_impeller_2"],
            n_blades_2=int(row["n_blades_2"]),
            pitch_angle_2=row["pitch_angle_2"],
            diameter_impeller_3=row["diameter_impeller_3"],
            ax_pos_impeller_3=row["ax_pos_impeller_3"],
            n_blades_3=int(row["n_blades_3"]),
            pitch_angle_3=row["pitch_angle_3"],
            n_diptubes=int(row["n_diptubes"]),
            n_baffles=int(row["n_baffles"]),
            rpm=row["rpm"],
            strainRate_p50=row["strainRate_p50"],
            epsilon_p50=row["epsilon_p50"],
            k_p50=row["k_p50"],
            Umag_p50=row["Umag_p50"],
        ))
    return rows
