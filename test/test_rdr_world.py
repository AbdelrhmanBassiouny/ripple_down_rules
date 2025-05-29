from __future__ import annotations

import sys
from unittest import TestCase

try:
    from PyQt6.QtWidgets import QApplication
    from ripple_down_rules.user_interface.gui import RDRCaseViewer
except ImportError as e:
    QApplication = None
    RDRCaseViewer = None

from conf.world.handles_and_containers import HandlesAndContainersWorld
from ripple_down_rules.datasets import *
from ripple_down_rules.datastructures.dataclasses import CaseQuery
from ripple_down_rules.experts import Human
from ripple_down_rules.helpers import is_matching
from ripple_down_rules.rdr import GeneralRDR
from ripple_down_rules.user_interface.gui import RDRCaseViewer


def get_all_possible_drawers(world: World) -> List[Drawer]:
    all_possible_drawers = []
    for handle in [body for body in world.bodies if isinstance(body, Handle)]:
        for container in [body for body in world.bodies if isinstance(body, Container)]:
            view = Drawer(handle, container, world=world)
            all_possible_drawers.append(view)
    return all_possible_drawers


class TestRDRWorld(TestCase):
    drawer_case_queries: List[CaseQuery]
    world: World
    app: Optional[QApplication] = None
    viewer: Optional[RDRCaseViewer] = None

    @classmethod
    def setUpClass(cls):
        cls.world = HandlesAndContainersWorld().create()
        cls.all_possible_drawers = get_all_possible_drawers(cls.world)

        print(cls.all_possible_drawers)
        cls.drawer_case_queries = [CaseQuery(possible_drawer, "correct", (bool,), True, default_value=False)
                                   for possible_drawer in cls.all_possible_drawers]

        if RDRCaseViewer is not None and QApplication is not None:
            cls.app = QApplication(sys.argv)
            cls.viewer = RDRCaseViewer(save_dir="./test_generated_rdrs")

    def test_view_rdr(self):
        self.get_view_rdr(use_loaded_answers=True, save_answers=False, append=False)

    def test_save_and_load_view_rdr(self):
        view_rdr = self.get_view_rdr(use_loaded_answers=True, save_answers=False, append=False)
        filename = os.path.join(os.getcwd(), "test_results/world_views_rdr")
        model_name = view_rdr.save(filename)
        loaded_rdr = GeneralRDR.load(filename, model_name=model_name)
        self.assertEqual(view_rdr.classify(self.world), loaded_rdr.classify(self.world))
        self.assertEqual(self.world.bodies, loaded_rdr.start_rules[0].corner_case.bodies)

    def test_write_view_rdr_to_python_file(self):
        rdrs_dir = "./test_generated_rdrs/view_rdr"
        os.makedirs(rdrs_dir, exist_ok=True)
        view_rdr = self.get_view_rdr()
        view_rdr._write_to_python(rdrs_dir)
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
        # print(found_views)
        for view in views:
            self.assertTrue(len([v for v in found_views["views"] if isinstance(v, view)]) > 0)

        return rdr

    def test_drawer_rdr(self):
        self.get_drawer_rdr(use_loaded_answers=True, save_answers=False)

    def test_write_drawer_rdr_to_python_file(self):
        rdrs_dir = "./test_generated_rdrs/drawer_rdr"
        os.makedirs(rdrs_dir, exist_ok=True)
        drawer_rdr = self.get_drawer_rdr()
        drawer_rdr._write_to_python(rdrs_dir)
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
