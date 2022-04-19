import copy
import random

from machetli.pddl import pddl_visitors
from machetli.successors import Successor, SuccessorGenerator


class RemoveActions(SuccessorGenerator):
    """Successor generator that removes 
    randomly selected actions from the PDDL task in a state.
    """
    def get_successors(self, state):
        """Yield modified versions of *state* of which in each
        one a different action is removed from the PDDL task
        stored in ``state["pddl_task"]``.
        """
        task = state["pddl_task"]
        action_names = [action.name for action in task.actions]
        random.Random().shuffle(action_names)
        for name in action_names:
            child_state = copy.deepcopy(state)
            pre_child_task = child_state["pddl_task"]
            child_state["pddl_task"] = pre_child_task.accept(
                pddl_visitors.TaskElementEraseActionVisitor(name))
            yield Successor(child_state,
                            f"removed 1 of {len(action_names)} actions.")


class ReplaceAtomsWithTruth(SuccessorGenerator):
    """Successor generator that removes 
    randomly selected atoms from the PDDL task in a state.
    This is accomplished by scanning the entire task for the
    atom to be removed, instantiating each instance of this atom
    with the *truth* value and then simplifying all logical expressions.
    """
    def get_successors(self, state):
        """Yield modified versions of *state* of which in each
        one a different atom is removed from the PDDL task
        stored in ``state["pddl_task"]``. 
        """
        task = state["pddl_task"]
        predicate_names = [predicate.name for predicate in task.predicates if
                           not (predicate.name == "dummy_axiom_trigger" or predicate.name == "=")]
        random.Random().shuffle(predicate_names)
        for name in predicate_names:
            child_state = copy.deepcopy(state)
            pre_child_task = child_state["pddl_task"]
            child_state["pddl_task"] = pre_child_task.accept(
                pddl_visitors.TaskElementErasePredicateTrueAtomVisitor(name))
            yield Successor(
                child_state,
                f"replaced 1 of {len(predicate_names)} atoms with Truth.")


class ReplaceAtomsWithFalsity(SuccessorGenerator):
    """Successor generator that removes 
    randomly selected atoms from the PDDL task in a state.
    The same mechanism is used as in :class:`ReplaceAtomsWithTruth <machetli.planning.generators.ReplaceAtomsWithTruth>`,
    but replacing atoms with *falsity* instead.
    """
    def get_successors(self, state):
        """Yield modified versions of *state* of which in each
        one a different atom is removed from the PDDL task
        stored in ``state["pddl_task"]``. 
        """
        task = state["pddl_task"]
        predicate_names = [predicate.name for predicate in task.predicates if
                           not (predicate.name == "dummy_axiom_trigger" or predicate.name == "=")]
        random.Random().shuffle(predicate_names)
        for name in predicate_names:
            child_state = copy.deepcopy(state)
            pre_child_task = child_state["pddl_task"]
            child_state["pddl_task"] = pre_child_task.accept(
                pddl_visitors.TaskElementErasePredicateFalseAtomVisitor(name))
            yield Successor(
                child_state,
                f"replaced 1 of {len(predicate_names)} atoms with Falsity.")


class ReplaceLiteralsWithTruth(SuccessorGenerator):
    def get_successors(self, state):
        task = state["pddl_task"]
        predicate_names = [predicate.name for predicate in task.predicates if
                           not (predicate.name == "dummy_axiom_trigger" or predicate.name == "=")]
        random.Random().shuffle(predicate_names)
        for name in predicate_names:
            child_state = copy.deepcopy(state)
            pre_child_task = child_state["pddl_task"]
            child_state["pddl_task"] = pre_child_task.accept(
                pddl_visitors.TaskElementErasePredicateTrueLiteralVisitor(name))
            yield Successor(
                child_state,
                f"replaced 1 of {len(predicate_names)} literals with Truth.")


class RemoveObjects(SuccessorGenerator):
    def get_successors(self, state):
        task = state["pddl_task"]
        object_names = [obj.name for obj in task.objects]
        random.Random().shuffle(object_names)
        for name in object_names:
            child_state = copy.deepcopy(state)
            pre_child_task = child_state["pddl_task"]
            child_state["pddl_task"] = pre_child_task.accept(
                pddl_visitors.TaskElementEraseObjectVisitor(name))
            yield Successor(child_state,
                            f"replaced 1 of {len(object_names)} objects.")
