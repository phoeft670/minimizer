import copy
import logging
import random

from machetli.pddl import visitors
from machetli.pddl.constants import KEY_IN_STATE
from machetli.successors import Successor, SuccessorGenerator

from machetli.tools import batched,ceildiv


class RemoveActions(SuccessorGenerator):
    """
    For each action schema in the PDDL domain, generate a successor
    where this action schema is removed. The order of the successors is
    randomized.
    """
    def get_successors(self, state):
        task = state[KEY_IN_STATE]
        action_names = [action.name for action in task.actions]
        random.Random().shuffle(action_names)
        for name in action_names:
            child_state = copy.deepcopy(state)
            pre_child_task = child_state[KEY_IN_STATE]
            child_state[KEY_IN_STATE] = pre_child_task.accept(
                visitors.TaskElementEraseActionVisitor(name))
            yield Successor(child_state,
                            f"Removed action '{name}'. Remaining actions: {len(task.actions) - 1}")


class RemovePredicates(SuccessorGenerator):
    """
    For each predicate in the PDDL domain, generate a successor where
    this predicate is compiled away. This is accomplished by scanning
    the entire task for the atom to be removed, instantiating each
    instance of this atom with a constant according to ``replace_with``:

    * ``"true"`` replaces all atoms of the removed predicate with true,
    * ``"false"`` replaces all atoms of the removed predicate with false, and
    * ``"dynamic"`` (default) replaces an atom of the removed predicate with
      true if it occurs positively and with false otherwise.

    The order of the successors is randomized.
    """
    def __init__(self, replace_with="dynamic"):
        self.replace_with = replace_with
        if replace_with == "dynamic":
            self.visitor = visitors.TaskElementErasePredicateTrueLiteralVisitor
        elif replace_with == "true":
            self.visitor = visitors.TaskElementErasePredicateTrueAtomVisitor
        elif replace_with == "false":
            self.visitor = visitors.TaskElementErasePredicateFalseAtomVisitor
        else:
            logging.critical(f"Used unknown option '{replace_with}' for "
                             f"replacing predicates.")

    def get_successors(self, state):
        task = state[KEY_IN_STATE]
        predicate_names = [predicate.name for predicate in task.predicates if
                           not (predicate.name == "dummy_axiom_trigger" or predicate.name == "=")]
        random.Random().shuffle(predicate_names)
        for name in predicate_names:
            child_state = copy.deepcopy(state)
            pre_child_task = child_state[KEY_IN_STATE]
            child_state[KEY_IN_STATE] = pre_child_task.accept(self.visitor(name))
            yield Successor(
                child_state,
                f"Removed predicate '{name}'. Remaining predicates: {len(task.predicates) - 1}")


class RemoveObjects(SuccessorGenerator):

    def __init__(self, batch_size=1, batches=None):
        self.batch_size = batch_size
        self.batches = batches

    """
    For each object in the PDDL problem, generate a successor that
    removes this object from the PDDL task. The order of the successors
    is randomized.
    """
    def get_successors(self, state):
        task = state[KEY_IN_STATE]
        object_names = [obj.name for obj in task.objects]
        random.Random().shuffle(object_names)
        if self.batches:
            self.batch_size = ceildiv(len(object_names), self.batches)
        print(f"RemoveObjects batch size: {self.batch_size} for {ceildiv(len(object_names), self.batch_size)} batches")
        for names in batched(object_names, self.batch_size):
            child_state = copy.deepcopy(state)
            pre_child_task = child_state[KEY_IN_STATE]
            child_state[KEY_IN_STATE] = pre_child_task.accept(
                visitors.TaskElementEraseObjectVisitor(names))
            yield Successor(child_state,
                            f"Remove objects '{names}'. Remaining objects: {len(task.objects) - len(names)}")

class RemoveObjectsAdaptive(SuccessorGenerator):

    def __init__(self, max_iterations=10000000):
        self.batch_size = 1
        self.iterations = 0
        self.max_iterations = max_iterations

    """
    For each object in the PDDL problem, generate a successor that
    removes this object from the PDDL task. The order of the successors
    is randomized.
    """
    def get_successors(self, state):
        task = state[KEY_IN_STATE]
        object_names = [obj.name for obj in task.objects]
        random.Random().shuffle(object_names)
        batches = min(len(object_names)//self.batch_size + 1, self.max_iterations)
        print(f"previous batches: {batches}")
        print(f"previous iterations: {self.iterations}")
        print(f"found successor after {(self.iterations/batches) * 100} percent")
        if self.iterations <= batches/5: # under 20% of batches
            self.batch_size += len(object_names)//25 # increase batch size by 4% of remaining operators
            print(f"new inc batch size: {self.batch_size}")
        elif self.iterations >= batches: # exhausted batches
            self.batch_size //= 4
            self.batch_size = max(self.batch_size, 1)
            print(f"new strongly dec batch size: {self.batch_size}")
        elif self.iterations > batches/2: # over 50% of batches
            self.batch_size //= 2
            self.batch_size = max(self.batch_size, 1)
            print(f"new dec batch size: {self.batch_size}")
        planned_batches = len(object_names)//self.batch_size + 1
        print(f"generating batches: {min(planned_batches, self.max_iterations)}/{planned_batches}")
        self.iterations = 0
        for names in batched(object_names, self.batch_size):
            if self.iterations >= self.max_iterations:
                break
            self.iterations += 1
            child_state = copy.deepcopy(state)
            pre_child_task = child_state[KEY_IN_STATE]
            child_state[KEY_IN_STATE] = pre_child_task.accept(
                visitors.TaskElementEraseObjectVisitor(names))
            message = (
                f"Remove objects '{names}'. Remaining objects: {len(task.objects) - len(names)}"
                if self.batch_size < 100
                else f"Remove {len(names)} objects. Remaining objects: {len(task.objects) - len(names)}"
            )
            yield Successor(child_state,message)
