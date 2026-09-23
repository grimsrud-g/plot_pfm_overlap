Author: Gracie Grimsrud
Date Created: 09/16/2026
Overall Project: PFM Compare
Analysis: PFM Output Overlap

Goal:

The scripts in this folder are designed to
	1. Take Template Matching PFMs from multiple processing streams/data types (rest vs task vs task residuals) for an individual as input
	2. Identify where network assignment (in TM, this is an integer) is the same across PFMs provided
	3. Create a new map (in .dscalar format and .png) that only has a network assignment for vertices that contained the SAME network
	assignment across all input
	4. Create a separate map for each input showing the network assignments that are unique to that input

Input Dir:/oak/stanford/groups/russpold/users/grimsrud/projects/pfm_compare/analysis/temp_match_results
Output Dir: /oak/stanford/groups/russpold/users/grimsrud/projects/pfm_compare/analysis/temp_match_overlap
	- 16Sept2026_test/network_overlap_outputs/sub-s19/   (single-subject notebook)
	- 22Sept2026_discovery/network_overlap_outputs/<subject>/ and .../group/   (all-subjects notebook)

Environment: network-fmri .venv (Python 3.12, nilearn 0.14)
	/oak/stanford/groups/russpold/users/grimsrud/projects/network-fmri-task/network-fmri/.venv/bin/python

Files:

1. temp_match_overlap_singlesub.ipynb
	Original step-by-step build for one subject (sub-s19): loads the rest / task / task_resid_raw dscalars,
	masks medial wall + unlabelled values (0, 4, 6, 17), finds consensus and per-input disagreement,
	saves dscalars + summary CSVs, and plots a surface figure (3 disagreement rows + consensus row)
	using network colors from the matching dlabel.

2. pfm_overlap.py
	Functions pulled from the single-subject notebook (same logic) so they can be reused across subjects:
	path building, label table/colormap, loading + validation, overlap, saving, and surface plotting.
	Checked against the single-subject notebook: sub-s19 outputs are identical.

3. temp_match_overlap_allsubs.ipynb
	Runs the same pipeline for the discovery sample (sub-s03, s10, s19, s29, s43) using pfm_overlap.py.
	Per subject: consensus + disagreement dscalars, summary CSVs, surface PNG.
	Group: stacked overall/network summary CSVs and a bar plot of consensus % per subject
	(consensus / grayordinates assigned in every input).
