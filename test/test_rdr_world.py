from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional
from unittest import TestCase

from ripple_down_rules.datastructures.dataclasses import CaseQuery
from ripple_down_rules.experts import Human
from ripple_down_rules.helpers import is_matching
from ripple_down_rules.rdr import GeneralRDR


@dataclass
class WorldEntity:
    world: World = field(kw_only=True, repr=False)


@dataclass
class Body(WorldEntity):
    name: str

    def __hash__(self):
        return hash(self.name)


@dataclass
class Handle(Body):
    ...


@dataclass
class Container(Body):
    ...


@dataclass
class Connection(WorldEntity):
    parent: Body
    child: Body


@dataclass
class FixedConnection(Connection):
    ...


@dataclass
class PrismaticConnection(Connection):
    ...


@dataclass
class World:
    bodies: List[Body] = field(default_factory=list)
    connections: List[Connection] = field(default_factory=list)
    views: List[View] = field(default_factory=list, repr=False)


@dataclass
class View(WorldEntity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.world.views.append(self)


@dataclass
class Drawer(View):
    handle: Handle
    container: Container
    correct: Optional[bool] = None

    def __hash__(self):
        return hash((self.handle.name, self.container.name))


@dataclass
class Cabinet(View):
    container: Container
    drawers: List[Drawer] = field(default_factory=list)

    def __hash__(self):
        return hash(tuple([self.container.name] + [hash(drawer) for drawer in self.drawers]))


class TestRDRWorld(TestCase):
    drawer_case_queries: List[CaseQuery]
    world: World

    @classmethod
    def setUpClass(cls):
        world = World()
        cls.world = world

        handle = Handle('h1', world=world)
        handle_2 = Handle('h2', world=world)
        container_1 = Container('c1', world=world)
        container_2 = Container('c2', world=world)
        connection_1 = FixedConnection(container_1, handle, world=world)
        connection_2 = PrismaticConnection(container_2, container_1, world=world)

        world.bodies = [handle, container_1, container_2, handle_2]
        world.connections = [connection_1, connection_2]

        all_possible_drawers = []
        for handle in [body for body in world.bodies if isinstance(body, Handle)]:
            for container in [body for body in world.bodies if isinstance(body, Container)]:
                view = Drawer(handle, container, world=world)
                all_possible_drawers.append(view)

        print(all_possible_drawers)
        cls.drawer_case_queries = [CaseQuery(possible_drawer, "correct", (bool,), True, default_value=False)
                                   for possible_drawer in all_possible_drawers]

    def test_view_rdr(self):
        self.get_view_rdr(use_loaded_answers=True, save_answers=False, append=False)

    def get_view_rdr(self, use_loaded_answers: bool = True, save_answers: bool = False,
                     append: bool = False):
        expert = Human(use_loaded_answers=use_loaded_answers)
        filename = os.path.join(os.getcwd(), "test_expert_answers/view_rdr_expert_answers_fit")
        if use_loaded_answers:
            expert.load_answers(filename)
        rdr = GeneralRDR()
        try:
            rdr.fit_case([CaseQuery(self.world, "views", (View,), False)], expert=expert,
                         add_extra_conclusions=True)
        except Exception as e:
            if append:
                expert.use_loaded_answers = False
                rdr.fit_case([CaseQuery(self.world, "views", (View,), False)], expert=expert,
                             add_extra_conclusions=True)
            else:
                raise e
        if save_answers:
            expert.save_answers(filename, append=append)
        print(rdr.classify(self.world))

    def test_drawer_rdr(self):
        self.get_drawer_rdr(use_loaded_answers=True, save_answers=False)

    def test_write_drawer_rdr_to_python_file(self):
        rdrs_dir = "./test_generated_rdrs"
        drawer_rdr = self.get_drawer_rdr()
        drawer_rdr.write_to_python_file(rdrs_dir)
        loaded_rdr_classifier = drawer_rdr.get_rdr_classifier_from_python_file(rdrs_dir)
        for case_query in self.drawer_case_queries:
            self.assertTrue(is_matching(loaded_rdr_classifier, case_query))

    def get_drawer_rdr(self, use_loaded_answers: bool = True, save_answers: bool = False):
        expert = Human(use_loaded_answers=use_loaded_answers)
        filename = os.path.join(os.getcwd(), "test_expert_answers/correct_drawer_rdr_expert_answers_fit")
        if use_loaded_answers:
            expert.load_answers(filename)
        rdr = GeneralRDR()
        rdr.fit(self.drawer_case_queries, expert=expert, animate_tree=False)
        if save_answers:
            expert.save_answers(filename)
        for case_query in self.drawer_case_queries:
            self.assertTrue(is_matching(rdr.classify, case_query))
        return rdr
