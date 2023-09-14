import pandas as pd
import scipy
from numpy.typing import NDArray


def load_data() -> pd.DataFrame:
    """Load the data."""
    # Path to the directory where the data (mat-file) is located
    path_cut_data = "/home/halli/ux/BMDLab/matlab/resprog/GUI/CleanCutsDL/"
    file_cut_data = "cutDataCinCTTI_rev_v2.mat"  # Name of the mat-file, containing the data

    mat_cut_data: NDArray = scipy.io.loadmat(
        path_cut_data + file_cut_data,
        simplify_cells=True,
    )["data"]

    df = pd.DataFrame(mat_cut_data)
    return df


# TODO: Er dette nødvendig?? Er dette bare tap av data? TODO: TEST!
def normalize_data_length(df: pd.DataFrame) -> pd.DataFrame:
    class_value_count = df["c_label"].value_counts().sort_index()
    classes = class_value_count.index
    print(f"Number of occurences of each consensus class: {class_value_count}")
    min_class_count = min(class_value_count.values)

    shortened = [df[df["c_label"] == class_val].sample(min_class_count) for class_val in classes]
    df_shortened = pd.concat(shortened, ignore_index=True)
    return df_shortened
