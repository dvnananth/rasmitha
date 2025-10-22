from reactorsim.data import load_stirred_reactor_cfd_csv
import pandas as pd
import tempfile, os


def test_load_stirred_reactor_cfd_csv():
    # Create a temporary CSV
    cols = [
        "volume","reactor_diameter","liquid_height","bottom_depth","n_impeller",
        "diameter_impeller_1","ax_pos_impeller_1","n_blades_1","pitch_angle_1",
        "diameter_impeller_2","ax_pos_impeller_2","n_blades_2","pitch_angle_2",
        "diameter_impeller_3","ax_pos_impeller_3","n_blades_3","pitch_angle_3",
        "n_diptubes","n_baffles","rpm","strainRate_p50","epsilon_p50","k_p50","Umag_p50",
    ]
    df = pd.DataFrame([[0.2238,0.1,0.015,0.02,1,0.035,-0.014,2,0,0,0,0,0,0,0,0,0,0,4,1000,36.0,0.04,0.0042,0.22]], columns=cols)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "tmp.csv")
        df.to_csv(path, index=False)
        rows = load_stirred_reactor_cfd_csv(path)
        assert len(rows) == 1
        assert rows[0].n_baffles == 4
