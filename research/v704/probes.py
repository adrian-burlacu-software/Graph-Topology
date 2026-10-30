"""Probes of the architecture, written for it to improve on by itself
(2026-10-30).

Two projects no model, reader or threshold has seen: `fixtures/rental`, which
the improving may look at (SEEN), and `fixtures/clinic`, which it may not
(HELD). Every family of capability is asked of both, in other words, so a
change that only fits its case gains on SEEN and nothing on HELD. A change is
credited with what it does to HELD and never chosen by it. Written whole
before any was run; a round changes none.

    python -m research.v704.converse --suite probes
    python -m research.v704.converse --suite probes-held
"""
from __future__ import annotations

from research.v704.converse import number, py, says, says_all, says_only, \
    stance, untouched_and_asked
from research.v704.suite import Conversation

IMPORT = "import sys; sys.path.insert(0, '.'); "
#: how an answer says a name is not there
NOT_THERE = ("no function", "not find", "couldn't find", "don't see",
             "doesn't exist", "does not exist", "no such", "isn't")


def R(area, turns, check):
    return Conversation(area, turns if isinstance(turns, list) else [turns],
                        check, True, "rental")


def C(area, turns, check):
    return Conversation(area, turns if isinstance(turns, list) else [turns],
                        check, True, "clinic")


SEEN = [
    # -- what the project holds, counted and listed ------------------------
    R("inventory", "how many functions are there in tools.py?", number(5)),
    R("inventory", "list the classes in hires.py",
      says_all("TooManyHires", "Hire")),
    R("inventory", "what constants does hires.py define?",
      says_all("HIRE_DAYS", "LATE_FEE", "MAX_HIRES")),
    # -- what is not there: said, not made up ------------------------------
    R("absent", "is there a function called refund?", stance("no")),
    R("absent", "what does the function sharpen_blade do?", says(*NOT_THERE)),
    R("absent", "does the project define a constant called MAX_TOOLS?",
      stance("no")),
    # -- a function run on what is said ------------------------------------
    R("run", "what does late_charge(3) return?", number(21)),
    R("run", "what does clean_code('a 9 b') give?", says("A9B")),
    R("run", "what would plain_cost(4) come to?", number(48)),
    # -- an edge of a function ---------------------------------------------
    R("edge", "what happens if dearest gets an empty list?",
      says("ValueError", "raises", "error")),
    R("edge", "what does late_charge do with a negative number of days?",
      says("ValueError", "raises")),
    R("edge", "what does of_group return when no tool is in the group?",
      says("empty", "[]", "no tools", "nothing")),
    # -- what a function raises --------------------------------------------
    R("raises", "which exception does hire raise?", says("TooManyHires")),
    R("raises", "when does hire refuse a customer?",
      says("MAX_HIRES", "4", "four", "as many")),
    R("raises", "does clean_code raise anything?", stance("no")),
    # -- who depends on what -----------------------------------------------
    R("depends", "which functions use LATE_FEE?", says("late_charge")),
    R("depends", "what does Hire.cost call?", says("deposit")),
    R("depends", "what uses read_tools?", says_all("page", "main")),
    # -- is it tested ------------------------------------------------------
    R("tested", "is late_charge tested?", stance("yes")),
    R("tested", "is there a test for dearest?", stance("no")),
    R("tested", "which functions in tools.py have tests?",
      says_only("clean_code", "dearest", "deposit", "of_group")),
    # -- data summed up ----------------------------------------------------
    R("data-sum", "what's the total value of all the tools in tools.csv?",
      number(1440)),
    R("data-sum", "how many customers are in customers.json?", number(4)),
    R("data-sum", "what is the average fee in the tools data?", number(19.8)),
    # -- data picked out ---------------------------------------------------
    R("data-pick", "which tools have a fee above 20?",
      says_all("Cement Mixer", "Pressure Washer")),
    R("data-pick", "who lives in Ashby?", says_all("Ines Moreau", "Wen Li")),
    R("data-pick", "which tool has the lowest value?",
      says_only("Long Ladder", "Cement Mixer", "Hammer Drill")),
    # -- what was just said ------------------------------------------------
    R("follow", ["where is LATE_FEE defined?", "and what is its value?"],
      number(7)),
    R("follow", ["what does deposit do?", "who calls it?"], says("cost")),
    R("follow", ["which tool is the dearest in the csv?",
                 "what group is it in?"], says("site")),
    # -- a small change ----------------------------------------------------
    R("change", "change MAX_HIRES to 6",
      py(IMPORT + "from rental.hires import MAX_HIRES as M; assert M == 6")),
    R("change", "rename late_charge to overdue_charge",
      py(IMPORT + "from rental.hires import overdue_charge as f; "
         "assert f(1) == 7")),
    R("change", "make dearest return None for an empty list",
      py(IMPORT + "from rental.tools import dearest as f, Tool; "
         "assert f([]) is None; "
         "assert f([Tool('a','a','a',1,1)]).code == 'a'")),
    # -- a change that cannot be placed: asked about, nothing touched ------
    R("unplaced", "make it faster", untouched_and_asked()),
    R("unplaced", "fix the bug", untouched_and_asked()),
    R("unplaced", "change the limit", untouched_and_asked()),
]

HELD = [
    C("inventory", "how many functions does patients.py have?", number(5)),
    C("inventory", "which classes are defined in visits.py?",
      says_all("DoctorFull", "Visit")),
    C("inventory", "name the constants in visits.py",
      says_all("VISIT_PRICE", "BOOK_AHEAD", "DAILY_LIMIT")),
    C("absent", "do we have a function named discharge?", stance("no")),
    C("absent", "what does prescribe_dose do?", says(*NOT_THERE)),
    C("absent", "is a constant named WARD_LIMIT defined anywhere?",
      stance("no")),
    C("run", "what does cancel_fee(30) return?", number(0)),
    C("run", "what do you get from tidy_number('P-7-1')?", says("P71")),
    C("run", "what would cancel_fee(2) come to?", number(15)),
    C("edge", "what happens when eldest is given an empty list?",
      says("ValueError", "raises", "error")),
    C("edge", "what does cancel_fee do with negative hours?",
      says("ValueError", "raises")),
    C("edge", "what does in_ward give when nobody is in that ward?",
      says("empty", "[]", "no patients", "nothing")),
    C("raises", "what exception can book raise?", says("DoctorFull")),
    C("raises", "when does book turn a patient away?",
      says("DAILY_LIMIT", "8", "eight", "no room")),
    C("raises", "can tidy_number raise an exception?", stance("no")),
    C("depends", "where is VISIT_PRICE used?", says("price_for")),
    C("depends", "which function does Visit.price call?", says("price_for")),
    C("depends", "what calls load_patients?", says_all("screen", "main")),
    C("tested", "does cancel_fee have a test?", stance("yes")),
    C("tested", "is eldest tested?", stance("no")),
    C("tested", "which functions of patients.py are covered by tests?",
      says_only("tidy_number", "eldest", "is_senior", "in_ward")),
    C("data-sum", "what do all the doctors' years add up to in doctors.csv?",
      number(41)),
    C("data-sum", "how many patients does patients.json hold?", number(6)),
    C("data-sum", "what's the mean age of the patients?", number(50.5)),
    C("data-pick", "which patients are older than 60?",
      says_all("Noor Haddad", "Kofi Mensah")),
    C("data-pick", "who is on the south ward?",
      says_all("Elio Rossi", "Aiko Sato")),
    C("data-pick", "which doctor has the fewest years?",
      says_only("Dr Bianchi", "Dr Amato", "Dr Chen")),
    C("follow", ["where is DAILY_LIMIT set?", "what's it set to?"], number(8)),
    C("follow", ["what does is_senior do?", "what calls it?"],
      says("price_for")),
    C("follow", ["who is the eldest patient in patients.json?",
                 "which ward are they on?"], says("north")),
    C("change", "set VISIT_PRICE to 55",
      py(IMPORT + "from clinic.visits import VISIT_PRICE as M; "
         "assert M == 55")),
    C("change", "rename cancel_fee to cancellation_fee",
      py(IMPORT + "from clinic.visits import cancellation_fee as f; "
         "assert f(2) == 15")),
    C("change", "make eldest return None when the list is empty",
      py(IMPORT + "from clinic.patients import eldest as f, Patient; "
         "assert f([]) is None; "
         "assert f([Patient('a','a',1,'a')]).name == 'a'")),
    C("unplaced", "make this better", untouched_and_asked()),
    C("unplaced", "there's a bug, fix it", untouched_and_asked()),
    C("unplaced", "update the price", untouched_and_asked()),
]
