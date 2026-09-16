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
Output Dir: TBD
