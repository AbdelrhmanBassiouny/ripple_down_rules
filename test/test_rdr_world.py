from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional
from unittest import TestCase

from PyQt6.QtWidgets import QApplication

from ripple_down_rules.datastructures.dataclasses import CaseQuery
from ripple_down_rules.experts import Human
from ripple_down_rules.gui import RDRCaseViewer
from ripple_down_rules.helpers import is_matching
from ripple_down_rules.rdr import GeneralRDR


@dataclass
class WorldEntity:
    world: Optional[World] = field(default=None, kw_only=True, repr=False, hash=False)


@dataclass(unsafe_hash=True)
class Body(WorldEntity):
    name: str


@dataclass(unsafe_hash=True)
class Handle(Body):
    ...


@dataclass(unsafe_hash=True)
class Container(Body):
    ...


@dataclass(unsafe_hash=True)
class Connection(WorldEntity):
    parent: Body
    child: Body


@dataclass(unsafe_hash=True)
class FixedConnection(Connection):
    ...


@dataclass(unsafe_hash=True)
class PrismaticConnection(Connection):
    ...


@dataclass
class World:
    id: int = 0
    bodies: List[Body] = field(default_factory=list)
    connections: List[Connection] = field(default_factory=list)
    views: List[View] = field(default_factory=list, repr=False)

    def __eq__(self, other):
        if not isinstance(other, World):
            return False
        return self.id == other.id


@dataclass(unsafe_hash=True)
class View(WorldEntity):
    ...


@dataclass(unsafe_hash=True)
class Drawer(View):
    handle: Handle
    container: Container
    correct: Optional[bool] = None


@dataclass
class Cabinet(View):
    container: Container
    drawers: List[Drawer] = field(default_factory=list)

    def __hash__(self):
        return hash((self.__class__.__name__, self.container))


class TestRDRWorld(TestCase):
    drawer_case_queries: List[CaseQuery]
    world: World
    app: QApplication
    viewer: RDRCaseViewer

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
        cls.app = QApplication(sys.argv)
        cls.viewer = RDRCaseViewer()

    def test_view_rdr(self):
        self.get_view_rdr(use_loaded_answers=False, save_answers=False, append=False)

    def test_save_and_load_view_rdr(self):
        view_rdr = self.get_view_rdr(use_loaded_answers=True, save_answers=False, append=False)
        filename = os.path.join(os.getcwd(), "test_results/world_views_rdr")
        view_rdr.save(filename)
        loaded_rdr = GeneralRDR.load(filename)
        self.assertEqual(view_rdr.classify(self.world), loaded_rdr.classify(self.world))
        self.assertEqual(self.world.bodies, loaded_rdr.start_rules[0].corner_case.bodies)

    def test_write_view_rdr_to_python_file(self):
        rdrs_dir = "./test_generated_rdrs"
        view_rdr = self.get_view_rdr()
        view_rdr.write_to_python_file(rdrs_dir)
        loaded_rdr_classifier = view_rdr.get_rdr_classifier_from_python_file(rdrs_dir)
        found_views = loaded_rdr_classifier(self.world)
        self.assertTrue(len([v for v in found_views["views"] if isinstance(v, Drawer)]) == 1)
        self.assertTrue(len([v for v in found_views["views"] if isinstance(v, Cabinet)]) == 1)
        self.assertTrue(len(found_views["views"]) == 2)

    def get_view_rdr(self, views=(Drawer, Cabinet), use_loaded_answers: bool = True, save_answers: bool = False,
                     append: bool = False):
        expert = Human(use_loaded_answers=use_loaded_answers, append=append, viewer=self.viewer)
        filename = os.path.join(os.getcwd(), "test_expert_answers/view_rdr_expert_answers_fit")
        if use_loaded_answers:
            expert.load_answers(filename)
        rdr = GeneralRDR()
        for view in views:
            rdr.fit_case(CaseQuery(self.world, "views", (view,), False), expert=expert)
        if save_answers:
            expert.save_answers(filename)

        found_views = rdr.classify(self.world)
        print(found_views)
        for view in views:
            self.assertTrue(len([v for v in found_views["views"] if isinstance(v, view)]) > 0)

        return rdr

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
