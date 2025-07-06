import importlib
import sys
from os.path import dirname

import pytest
from typing_extensions import Callable

try:
    from PyQt6.QtWidgets import QApplication
    from ripple_down_rules.user_interface.gui import RDRCaseViewer
except ImportError as e:
    QApplication = None
    RDRCaseViewer = None

from ..conf.world.handles_and_containers import HandlesAndContainersWorld
from ..datasets import *
from ripple_down_rules.datastructures.dataclasses import CaseQuery
from ripple_down_rules.experts import Human
from ripple_down_rules.helpers import is_matching
from ripple_down_rules.rdr import GeneralRDR

app: Optional[QApplication] = None
viewer: Optional[RDRCaseViewer] = None
use_gui: bool = False

if RDRCaseViewer is not None and QApplication is not None and use_gui:
    app = QApplication(sys.argv)
    viewer = RDRCaseViewer(save_dir="./test_generated_rdrs")


def handles_and_containers_world() -> World:
    return HandlesAndContainersWorld().create()

def get_possible_drawers() -> List[Drawer]:
    """
    Get all possible drawer types.
    """
    all_possible_drawers = []
    world = handles_and_containers_world()
    for handle in [body for body in world.bodies if isinstance(body, Handle)]:
        for container in [body for body in world.bodies if isinstance(body, Container)]:
            view = Drawer(handle, container, world=world)
            all_possible_drawers.append(view)
    return all_possible_drawers

@pytest.fixture
def drawer_case_queries() -> List[CaseQuery]:
    case_queries = [CaseQuery(possible_drawer, "correct", (bool,), True, default_value=False,
                              case_factory=get_possible_drawers, case_factory_idx=i)
                    for i, possible_drawer in enumerate(get_possible_drawers())]
    return case_queries


@dataclass
class ExpertConfig:
    filename: str
    use_loaded_answers: bool


@pytest.fixture
def expert():
    def _create(conf: ExpertConfig):
        """
        Fixture to create an expert.

        :param conf: ExpertConfig object containing configuration for the expert.
        """
        human = Human(use_loaded_answers=conf.use_loaded_answers)
        filename = os.path.join(dirname(__file__), "../test_expert_answers", conf.filename)
        human.load_answers(filename)
        return human

    return _create


@pytest.fixture
def drawer_cabinet_expert(expert) -> Human:
    """
    Fixture to create an expert for drawer and cabinet views.
    """
    conf = ExpertConfig("drawer_cabinet_expert_answers_fit", use_loaded_answers=True)
    return expert(conf)


@pytest.fixture
def drawer_expert(expert) -> Human:
    """
    Fixture to create an expert for drawer views.
    """
    conf = ExpertConfig("correct_drawer_rdr_expert_answers_fit", use_loaded_answers=True)
    return expert(conf)


@pytest.fixture
def drawer_case_query() -> CaseQuery:
    """
    Create a CaseQuery for the given view type in the provided world.
    """
    return CaseQuery(handles_and_containers_world(), "views", (Drawer,), False,
                     case_factory=handles_and_containers_world)


def possibilities_rdr_verification(rdr: GeneralRDR, case_query: CaseQuery) -> None:
    conclusions = rdr.classify(case_query.original_case)[case_query.attribute_name]
    assert len([v for v in conclusions if isinstance(v, case_query.core_attribute_type)]) > 0


@pytest.fixture
def drawer_rdr(drawer_case_query, drawer_cabinet_expert) -> GeneralRDR:
    """
    Fixture to create a GeneralRDR for drawer views.
    """
    rdr = GeneralRDR()
    rdr.fit_case(drawer_case_query, expert=drawer_cabinet_expert)
    possibilities_rdr_verification(rdr, drawer_case_query)
    return rdr

def get_method_object_from_pytest_request(request) -> Callable:
    test_module = request.module.__name__  # e.g., "test_my_module"
    test_class = request.cls.__name__ if request.cls else None  # if inside a class
    test_name = request.node.name
    func = importlib.import_module(test_module)
    if test_class:
        func = getattr(getattr(func, test_class), test_name)
    else:
        func = getattr(func, test_name)
    return func

@pytest.fixture
def drawer_cabinet_rdr(request, drawer_cabinet_expert) -> GeneralRDR:
    world = handles_and_containers_world()
    rdr = GeneralRDR()
    for view in [Drawer, Cabinet]:
        rdr.fit_case(CaseQuery(world, "views", (view,), False, case_factory=handles_and_containers_world),
                     expert=drawer_cabinet_expert, scenario=get_method_object_from_pytest_request(request))
    found_views = rdr.classify(world)
    for view in [Drawer, Cabinet]:
        assert len([v for v in found_views["views"] if isinstance(v, view)]) > 0
    return rdr


@pytest.fixture
def correct_drawer_rdr(drawer_case_queries, drawer_expert) -> GeneralRDR:
    rdr = GeneralRDR()
    rdr.fit(drawer_case_queries, expert=drawer_expert, animate_tree=False)
    for case_query in drawer_case_queries:
        assert is_matching(rdr.classify, case_query)
    return rdr
