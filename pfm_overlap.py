"""
Reusable functions for comparing template-matching PFMs across inputs
(e.g., rest vs. task vs. task residuals) within a subject.

Extracted from temp_match_overlap_singlesub.ipynb so that the
single-subject and multi-subject notebooks share one implementation.

Author: Gracie Grimsrud
Date Created: 09/22/2026
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from nibabel.cifti2 import Cifti2Header
from nibabel.cifti2.cifti2_axes import BrainModelAxis, LabelAxis, ScalarAxis
from nilearn import plotting


MEDIAL_WALL_VALUE = 0
UNLABELLED_VALUES = (4, 6, 17)
BACKGROUND_VALUES = {MEDIAL_WALL_VALUE, *UNLABELLED_VALUES}
INTEGER_TOLERANCE = 1e-6


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

def build_dscalar_path(
    base_directory,
    results_directory,
    subject,
    session,
    task,
    common_suffix,
):
    filename = (
        f"{subject}_{session}_task-{task}_"
        f"{common_suffix}"
    )

    return (
        Path(base_directory)
        / results_directory
        / subject
        / f"task-{task}"
        / subject
        / session
        / "func"
        / filename
    )


def build_input_files(
    base_directory,
    input_config,
    subject,
    session,
    common_suffix,
):
    """
    Return {input_name: dscalar_path} for one subject.
    """
    return {
        input_name: build_dscalar_path(
            base_directory=base_directory,
            results_directory=config["results_directory"],
            subject=subject,
            session=session,
            task=config["task"],
            common_suffix=common_suffix,
        )
        for input_name, config in input_config.items()
    }


def build_splithalf_dscalar_path(
    base_directory,
    subject,
    session,
    task,
    common_suffix,
):
    """
    Split-half outputs use a flatter layout than build_dscalar_path:
    {base_directory}/{subject}/{session}/func/{filename}
    """
    filename = (
        f"{subject}_{session}_task-{task}_"
        f"{common_suffix}"
    )

    return (
        Path(base_directory)
        / subject
        / session
        / "func"
        / filename
    )


def build_splithalf_input_files(
    base_directory,
    subject,
    task,
    sessions,
    common_suffix,
):
    """
    Return {half_name: dscalar_path} for one subject and task, where
    sessions is {half_name: session}, e.g. {"half1": "ses-splithalf1"}.
    """
    return {
        half_name: build_splithalf_dscalar_path(
            base_directory=base_directory,
            subject=subject,
            session=session,
            task=task,
            common_suffix=common_suffix,
        )
        for half_name, session in sessions.items()
    }


def corresponding_dlabel_path(dscalar_path):
    dscalar_path = Path(dscalar_path)

    expected_suffix = ".dscalar.nii"

    if not dscalar_path.name.endswith(expected_suffix):
        raise ValueError(
            f"Expected a filename ending in {expected_suffix}: "
            f"{dscalar_path}"
        )

    dlabel_name = dscalar_path.name.replace(
        ".dscalar.nii",
        ".dlabel.nii",
    )

    return dscalar_path.with_name(dlabel_name)


def safe_filename_component(text):
    """
    Replace characters unsuitable for filenames.
    """
    cleaned = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        str(text),
    ).strip("._")

    if not cleaned:
        raise ValueError(
            f"Could not make a safe filename from {text!r}"
        )

    return cleaned


# ---------------------------------------------------------------------
# Network labels and colors
# ---------------------------------------------------------------------

def load_label_table(dlabel_path, map_index=0):
    """
    Return (network_labels, network_colors) from a dlabel file:
    {integer: name} and {integer: (r, g, b, a)}.
    """
    dlabel_img = nib.load(dlabel_path)
    dlabel_axis = dlabel_img.header.get_axis(0)

    if not isinstance(dlabel_axis, LabelAxis):
        raise TypeError(
            f"Expected a LabelAxis in {dlabel_path}, "
            f"found {type(dlabel_axis)}"
        )

    label_table = dlabel_axis.label[map_index]

    network_labels = {
        int(integer_value): label_information[0]
        for integer_value, label_information
        in label_table.items()
    }

    network_colors = {
        int(integer_value): tuple(label_information[1])
        for integer_value, label_information
        in label_table.items()
    }

    return network_labels, network_colors


def make_network_cmap(
    network_colors,
    background_values=BACKGROUND_VALUES,
):
    """
    Build a ListedColormap with one slot per label integer, so that
    label k maps to slot k when vmin=-0.5 and vmax=max_label+0.5.

    Returns (cmap, color_min, color_max).
    """
    maximum_label_id = max(network_colors)

    color_array = np.zeros(
        (maximum_label_id + 1, 4),
        dtype=float,
    )

    # Unused slots default to opaque gray.
    color_array[:] = (0.75, 0.75, 0.75, 1.0)

    # dlabel colors, forced to full opacity.
    for network_id, rgba in network_colors.items():
        opaque_rgba = list(rgba)
        opaque_rgba[3] = 1.0

        color_array[network_id] = opaque_rgba

    # Excluded values are transparent.
    for background_id in background_values:
        if background_id < color_array.shape[0]:
            color_array[background_id] = (1.0, 1.0, 1.0, 0.0)

    cmap = ListedColormap(
        color_array,
        name="dlabel_network_colors",
    )

    color_min = -0.5
    color_max = maximum_label_id + 0.5

    return cmap, color_min, color_max


# ---------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------

def make_background_mask(
    values,
    background_values=BACKGROUND_VALUES,
    tolerance=INTEGER_TOLERANCE,
):
    """
    Return True for medial-wall, unlabelled, NaN,
    and infinite values.
    """
    mask = ~np.isfinite(values)

    for background_code in background_values:
        mask |= np.isclose(
            values,
            background_code,
            atol=tolerance,
            rtol=0,
        )

    return mask


def same_brain_model(axis_a, axis_b):
    if not isinstance(axis_a, BrainModelAxis):
        return False

    if not isinstance(axis_b, BrainModelAxis):
        return False

    if axis_a.size != axis_b.size:
        return False

    if not np.array_equal(axis_a.name, axis_b.name):
        return False

    if not np.array_equal(axis_a.vertex, axis_b.vertex):
        return False

    if not np.array_equal(axis_a.voxel, axis_b.voxel):
        return False

    if axis_a.nvertices != axis_b.nvertices:
        return False

    if axis_a.volume_shape != axis_b.volume_shape:
        return False

    if axis_a.affine is None or axis_b.affine is None:
        return (
            axis_a.affine is None
            and axis_b.affine is None
        )

    return np.allclose(axis_a.affine, axis_b.affine)


def load_assignments(
    input_files,
    map_index=0,
    background_values=BACKGROUND_VALUES,
    tolerance=INTEGER_TOLERANCE,
):
    """
    Load one map from each dscalar and validate that all inputs share
    the same brain model and contain integer-coded assignments.

    Returns a dict with:
        input_names   list of input names (row order)
        assignments   (n_inputs, n_grayordinates) float array
        background    (n_inputs, n_grayordinates) bool array
        reference_img first input's Cifti2Image
        reference_axis first input's BrainModelAxis
    """
    input_names = list(input_files)

    images = {}
    assignments_by_input = {}
    background_masks_by_input = {}
    brain_axes = {}

    for input_name, input_path in input_files.items():
        if not Path(input_path).is_file():
            raise FileNotFoundError(
                f"Missing input:\n{input_path}"
            )

        img = nib.load(input_path)

        if not isinstance(img, nib.Cifti2Image):
            raise TypeError(
                f"{input_name} is not a CIFTI-2 image"
            )

        if not 0 <= map_index < img.shape[0]:
            raise IndexError(
                f"{input_name} contains {img.shape[0]} map(s), "
                f"so map_index={map_index} is invalid"
            )

        values = np.asanyarray(
            img.dataobj[map_index],
            dtype=np.float64,
        ).reshape(-1)

        background_mask = make_background_mask(
            values,
            background_values,
            tolerance,
        )

        # Confirm retained values are integers, then normalize
        # tiny floating-point deviations.
        assigned_values = values[~background_mask]
        rounded_values = np.rint(assigned_values)

        integer_matches = np.isclose(
            assigned_values,
            rounded_values,
            atol=tolerance,
            rtol=0,
        )

        if not np.all(integer_matches):
            raise ValueError(
                f"{input_name} contains non-integer "
                f"assignments. Examples: "
                f"{assigned_values[~integer_matches][:10]}"
            )

        values[~background_mask] = rounded_values

        images[input_name] = img
        assignments_by_input[input_name] = values
        background_masks_by_input[input_name] = background_mask
        brain_axes[input_name] = img.header.get_axis(1)

    reference_name = input_names[0]
    reference_axis = brain_axes[reference_name]

    for input_name, brain_axis in brain_axes.items():
        if not same_brain_model(reference_axis, brain_axis):
            raise ValueError(
                f"{input_name} does not use the same "
                f"CIFTI brain-model axis as {reference_name}"
            )

    assignments = np.vstack([
        assignments_by_input[input_name]
        for input_name in input_names
    ])

    background = np.vstack([
        background_masks_by_input[input_name]
        for input_name in input_names
    ])

    return {
        "input_names": input_names,
        "assignments": assignments,
        "background": background,
        "reference_img": images[reference_name],
        "reference_axis": reference_axis,
    }


# ---------------------------------------------------------------------
# Overlap
# ---------------------------------------------------------------------

def compute_overlap(
    assignments,
    background,
    input_names,
    network_labels,
):
    """
    Identify grayordinates with the same network in every input
    (consensus) and, for each input, its assignments wherever the
    inputs do not all agree (disagreement).

    Returns a dict with:
        all_assigned        bool, assigned in every input
        consensus_mask      bool, assigned everywhere and same network
        disagreement_mask   bool, assigned somewhere and not consensus
        consensus_values    float32 map of consensus network IDs (0 elsewhere)
        disagreement_maps   {input_name: float32 map}
        overall_summary     DataFrame of grayordinate category counts
        network_summary     DataFrame of per-network counts
    """
    all_assigned = (~background).all(axis=0)

    same_network = (
        assignments == assignments[0]
    ).all(axis=0)

    consensus_mask = all_assigned & same_network

    has_any_assignment = (~background).any(axis=0)

    disagreement_mask = (
        has_any_assignment
        & ~consensus_mask
    )

    n_grayordinates = assignments.shape[1]

    consensus_values = np.zeros(
        n_grayordinates,
        dtype=np.float32,
    )

    consensus_values[consensus_mask] = assignments[
        0,
        consensus_mask,
    ]

    disagreement_maps = {}

    for row_index, input_name in enumerate(input_names):
        output_values = np.zeros(
            n_grayordinates,
            dtype=np.float32,
        )

        keep = (
            disagreement_mask
            & ~background[row_index]
        )

        output_values[keep] = assignments[row_index, keep]

        disagreement_maps[input_name] = output_values

    overall_summary = pd.DataFrame({
        "category": [
            "Total grayordinates",
            "Valid assignment in every input",
            "Consensus assignment",
            "Disagreement with at least one valid assignment",
            "Background or unlabelled in every input",
        ],
        "count": [
            n_grayordinates,
            int(all_assigned.sum()),
            int(consensus_mask.sum()),
            int(disagreement_mask.sum()),
            int((~has_any_assignment).sum()),
        ],
    })

    overall_summary["percent_of_all_grayordinates"] = (
        100
        * overall_summary["count"]
        / n_grayordinates
    )

    valid_network_ids = set()

    for row_index in range(len(input_names)):
        valid_values = assignments[
            row_index,
            ~background[row_index],
        ]

        valid_network_ids.update(
            int(value)
            for value in np.unique(valid_values)
        )

    network_summary_rows = []

    for network_id in sorted(valid_network_ids):
        row = {
            "network_id": network_id,
            "network_name": network_labels.get(
                network_id,
                "Unknown label",
            ),
            "consensus_count": int(
                np.sum(consensus_values == network_id)
            ),
        }

        for input_name in input_names:
            row[f"{input_name}_disagreement_count"] = int(
                np.sum(
                    disagreement_maps[input_name]
                    == network_id
                )
            )

        network_summary_rows.append(row)

    network_summary = pd.DataFrame(network_summary_rows)

    return {
        "all_assigned": all_assigned,
        "consensus_mask": consensus_mask,
        "disagreement_mask": disagreement_mask,
        "consensus_values": consensus_values,
        "disagreement_maps": disagreement_maps,
        "overall_summary": overall_summary,
        "network_summary": network_summary,
    }


# ---------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------

def save_dscalar(
    values,
    map_name,
    output_path,
    reference_img,
    reference_axis,
    overwrite=False,
):
    """
    Save one one-dimensional grayordinate array as
    a single-map CIFTI dscalar file.
    """
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite:\n{output_path}\n"
            "Pass overwrite=True if replacement is intended."
        )

    values = np.asarray(
        values,
        dtype=np.float32,
    ).reshape(-1)

    if values.size != reference_axis.size:
        raise ValueError(
            f"Output contains {values.size} values, "
            f"but the reference brain axis contains "
            f"{reference_axis.size} grayordinates."
        )

    output_header = Cifti2Header.from_axes((
        ScalarAxis([map_name]),
        reference_axis,
    ))

    output_img = nib.Cifti2Image(
        values[np.newaxis, :],
        header=output_header,
        nifti_header=reference_img.nifti_header.copy(),
    )

    output_img.update_headers()

    nib.save(output_img, output_path)

    return output_path


def save_overlap_outputs(
    overlap,
    loaded,
    subject,
    comparison_name,
    output_directory,
    overwrite=False,
):
    """
    Save consensus + per-input disagreement dscalars and the two
    summary CSVs for one subject.

    Returns a dict of output paths.
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    prefix = f"{subject}_{comparison_name}"

    consensus_path = save_dscalar(
        values=overlap["consensus_values"],
        map_name="consensus_all_inputs",
        output_path=output_directory / f"{prefix}_consensus.dscalar.nii",
        reference_img=loaded["reference_img"],
        reference_axis=loaded["reference_axis"],
        overwrite=overwrite,
    )

    disagreement_paths = {}

    for input_name, values in overlap["disagreement_maps"].items():
        clean_input_name = safe_filename_component(input_name)

        disagreement_paths[input_name] = save_dscalar(
            values=values,
            map_name=f"disagreement_{clean_input_name}",
            output_path=(
                output_directory
                / f"{prefix}_disagreement-{clean_input_name}.dscalar.nii"
            ),
            reference_img=loaded["reference_img"],
            reference_axis=loaded["reference_axis"],
            overwrite=overwrite,
        )

    overall_summary_path = output_directory / f"{prefix}_overall_summary.csv"
    network_summary_path = output_directory / f"{prefix}_network_summary.csv"

    for path in (overall_summary_path, network_summary_path):
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite:\n{path}\n"
                "Pass overwrite=True if replacement is intended."
            )

    overlap["overall_summary"].to_csv(overall_summary_path, index=False)
    overlap["network_summary"].to_csv(network_summary_path, index=False)

    return {
        "consensus_path": consensus_path,
        "disagreement_paths": disagreement_paths,
        "overall_summary_path": overall_summary_path,
        "network_summary_path": network_summary_path,
    }


# ---------------------------------------------------------------------
# Surface plotting
# ---------------------------------------------------------------------

def extract_cortical_values(cifti_img, map_index=0):
    """
    Extract left- and right-cortical values from one
    map in a CIFTI-2 image.
    """
    brain_axis = cifti_img.header.get_axis(1)

    values = np.asanyarray(
        cifti_img.dataobj[map_index],
        dtype=np.float64,
    ).reshape(-1)

    left_structure = "CIFTI_STRUCTURE_CORTEX_LEFT"
    right_structure = "CIFTI_STRUCTURE_CORTEX_RIGHT"

    left_values = np.zeros(
        brain_axis.nvertices[left_structure],
        dtype=np.float64,
    )

    right_values = np.zeros(
        brain_axis.nvertices[right_structure],
        dtype=np.float64,
    )

    found_left = False
    found_right = False

    for (
        structure_name,
        data_indices,
        structure_axis,
    ) in brain_axis.iter_structures():

        if structure_name == left_structure:
            left_values[structure_axis.vertex] = values[data_indices]
            found_left = True

        elif structure_name == right_structure:
            right_values[structure_axis.vertex] = values[data_indices]
            found_right = True

    if not found_left:
        raise ValueError(
            "The CIFTI does not contain left-cortical data."
        )

    if not found_right:
        raise ValueError(
            "The CIFTI does not contain right-cortical data."
        )

    return left_values, right_values


def zero_background(
    values,
    background_values=BACKGROUND_VALUES,
    tolerance=INTEGER_TOLERANCE,
):
    """
    Set medial-wall, unlabelled, and NaN values to 0 so
    plot_surf_roi thresholds them out and shows the plain surface.
    """
    values = values.copy()

    values[
        make_background_mask(
            values,
            background_values,
            tolerance,
        )
    ] = 0

    return values


def load_hemisphere_values(maps_to_plot, map_index=0):
    """
    Return {map_label: {"left": array, "right": array}} with background
    zeroed, plus the sorted network IDs present across all maps.
    """
    hemisphere_values_by_map = {}
    plotted_network_ids = set()

    for map_label, map_path in maps_to_plot.items():
        left_values, right_values = extract_cortical_values(
            nib.load(map_path),
            map_index,
        )

        hemisphere_values_by_map[map_label] = {
            "left": zero_background(left_values),
            "right": zero_background(right_values),
        }

        for hemi_values in hemisphere_values_by_map[map_label].values():
            plotted_network_ids.update(
                int(value)
                for value in np.unique(hemi_values)
                if value != 0
            )

    return hemisphere_values_by_map, sorted(plotted_network_ids)


def plot_surface_grid(
    hemisphere_values_by_map,
    plotted_network_ids,
    surface_views,
    cmap,
    color_min,
    color_max,
    network_labels,
    network_colors,
    title=None,
    panel_zoom=1.5,
    legend_fontsize=13,
):
    """
    One row per map, one column per (hemi, view, surface_path) in
    surface_views, plus a legend of the networks present.

    Returns the matplotlib Figure (closed, so it is not auto-displayed).
    """
    map_labels = list(hemisphere_values_by_map)

    fig, axes = plt.subplots(
        len(map_labels),
        len(surface_views),
        figsize=(
            4 * len(surface_views),
            2.6 * len(map_labels),  # short rows to match wide brains
        ),
        subplot_kw={"projection": "3d"},
        squeeze=False,
        # ipykernel's inline backend defaults to a transparent facecolor
        facecolor="white",
    )

    for row_index, map_label in enumerate(map_labels):
        for column_index, (hemi, view, surface_path) in enumerate(
            surface_views
        ):
            ax = axes[row_index, column_index]

            plotting.plot_surf_roi(
                str(surface_path),
                roi_map=hemisphere_values_by_map[map_label][hemi],
                hemi=hemi,
                view=view,
                cmap=cmap,
                vmin=color_min,
                vmax=color_max,
                threshold=1e-14,
                colorbar=False,
                axes=ax,
                figure=fig,
            )

            # must come after plot_surf_roi, which sets zoom=1.3
            ax.set_box_aspect(None, zoom=panel_zoom)

            if row_index == 0:
                ax.set_title(f"{hemi} {view}", fontsize=13)

        axes[row_index, 0].text2D(
            -0.08,
            0.5,
            map_label,
            transform=axes[row_index, 0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=14,
        )

    legend_handles = [
        Patch(
            facecolor=network_colors[network_id][:3],
            edgecolor="gray",
            label=f"{network_id}: {network_labels[network_id]}",
        )
        for network_id in plotted_network_ids
    ]

    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.85, 0.5),  # figure coords, just right of the grid
        frameon=False,
        fontsize=legend_fontsize,
    )

    # fixed margins in inches so the title clears the column titles
    # for any number of rows
    figure_height = fig.get_figheight()

    if title is not None:
        fig.suptitle(title, fontsize=16, y=1 - 0.15 / figure_height)

    # tight_layout handles 3D axes poorly; set margins/gaps manually.
    # Negative wspace/hspace overlap the mostly empty 3D panel boxes.
    fig.subplots_adjust(
        left=0.03,
        right=0.84,  # leave room for the legend
        bottom=0.0,
        top=1 - 0.75 / figure_height,
        wspace=-0.05,
        hspace=-0.1,
    )

    plt.close(fig)

    return fig
