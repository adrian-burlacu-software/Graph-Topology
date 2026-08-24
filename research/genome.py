GENOME = {
    "growth": {
        "max_children_per_cell": 2,
        "growth_threshold": 0.55,
        "growth_cost": 1.0
    },

    "connection": {
        "initial_strength": 0.5,
        "learning_rate": 0.08,
        "max_connections_per_cell": 8
    },

    "ordering": {
        "decay": 0.90,
        "sequence_strength": 1.0
    },

    "inhibition": {
        "strength": 0.35,
        "radius": 1
    },

    "reuse": {
        "match_threshold": 0.70,
        "reuse_reward": 1.0
    },

    "pruning": {
        "minimum_strength": 0.05,
        "minimum_activity": 0.01
    }
}