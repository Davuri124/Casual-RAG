# v1.4 local cliff test (GPU T4, 125 min)
Code 1.4.0. 150 applicants (first 150 of the 600 evidence bundles) x 4 clauses = 595 valid pairs per system (5 skipped: loan amount would be clipped).
Systems: no_rag, causal_conditions_only, causal_rag_smooth, causal_graded.
cliff_index_abs = mean|P(decline) change across the cutoff| / mean|change over an equal-width interval away from it|  (1 = smooth).
no_rag 1.08 [0.81, 1.49] | graded 2.76 [2.21, 3.46] | conditions-only 5.29 [4.00, 7.00] | smooth 86.0 [53.3, 172.9]
Source: user's notebook45a5072343.ipynb outputs.
