from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional
from unittest import TestCase

from ripple_down_rules.datastructures.dataclasses import CaseQuery
from ripple_down_rules.experts import Human
from ripple_down_rules.rdr import SingleClassRDR, GeneralRDR


@dataclass
class WorldEntity:
    world: Optional[World] = field(kw_only=True, default=None, repr=False)


@dataclass
class Body(WorldEntity):
    name: str


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


@dataclass
class View(WorldEntity):
    ...


@dataclass
class Drawer(View):
    handle: Handle
    container: Container
    correct: Optional[bool] = None


class TestRDRWorld(TestCase):
    drawer_case_queries: List[CaseQuery]

    @classmethod
    def setUpClass(cls):
        world = World()

        handle = Handle('h1', world=world)
        handle_2 = Handle('h2', world=world)
        container_1 = Container('c1',world=world)
        container_2 = Container('c2', world=world)
        connection_1 = FixedConnection(container_1, handle, world=world)
        connection_2 = PrismaticConnection(container_2, container_1, world=world)

        world.bodies = [handle, container_1, container_2, handle_2]
        world.connections = [connection_1, connection_2]

        all_views = []

        i = 1
        for handle in [body for body in world.bodies if isinstance(body, Handle)]:
            for container in [body for body in world.bodies if isinstance(body, Container)]:
                view = Drawer(handle, container, world=world)
                all_views.append(view)
                i += 1

        print(all_views)
        cls.drawer_case_queries = [CaseQuery(view, "correct", bool, True, default_value=False) for view in all_views]

    def test_drawer_scrdr(self):
        use_loaded_answers = True
        save_answers = False
        expert = Human(use_loaded_answers=use_loaded_answers)
        filename = os.path.join(os.getcwd(), "test_expert_answers/scrdr_world_expert_answers_fit")
        if use_loaded_answers:
            expert.load_answers(filename)
        rdr = SingleClassRDR()
        rdr.fit(self.drawer_case_queries, expert=expert, animate_tree=False)
        if save_answers:
            expert.save_answers(filename)
        for case_query in self.drawer_case_queries:
            self.assertEqual(rdr.classify(case_query.case), case_query.target_value)
            # print(f"Case: {case_query}, Classification: {rdr.classify(case_query.case)}")
