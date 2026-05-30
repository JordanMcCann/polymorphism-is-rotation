"""polymorphism-is-rotation: replication code for the paper of the same name.

Public surface:
    polymorphism.model     -- 2-layer transformer (104k params, Dyck-3)
    polymorphism.task      -- bounded-depth Dyck-3 generator + labels
    polymorphism.train     -- AdamW + cosine schedule, bf16
    polymorphism.symmetry_search -- group alignment for Bar P
    polymorphism.rmsnorm_fold    -- analytical RMSNorm folding
    polymorphism.analysis  -- the five lenses (weights, SAEs, causal, polyhedral, constructive)
    polymorphism.verification    -- the four bars (B, P, C, Pr) + adversarial decoy
    polymorphism.experiments     -- cross-seed, cross-checkpoint, scale (Pythia-70m), etc.
"""

__version__ = "1.0.0"
