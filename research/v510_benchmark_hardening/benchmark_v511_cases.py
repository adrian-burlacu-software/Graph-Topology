"""V511 attention-trajectory cases.

V511 intentionally reuses the frozen V509/V510 task set.  It adds no new
knowledge and does not change correctness criteria.  The benchmark's new
purpose is to expose the planner/attention trajectory, especially when the
conversation target changes.
"""

from benchmark_cases import CASES as V509_CASES
from benchmark_v510_cases import V510_ADVERSARIAL_CASES

V511_CASES = V509_CASES + V510_ADVERSARIAL_CASES

SWITCH_PROBES = [
    {"name":"dog_property_switch", "turns":["the dog is red","what color is the dog?","what size is the dog?"], "focus":"switch/property"},
    {"name":"subject_switch", "turns":["the dog is red","the cat is blue","what color is the cat?"], "focus":"switch/subject"},
    {"name":"subject_property_switch", "turns":["the dog is red","the cat is blue","what size is the cat?"], "focus":"switch/subject+property"},
    {"name":"return_to_previous_subject", "turns":["the dog is red","the cat is blue","what color is the dog?"], "focus":"switch/return"},
]
