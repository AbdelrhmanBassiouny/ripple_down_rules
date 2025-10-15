"""
Cell component classes for building Jupyter notebooks programmatically.

This module provides an OOP architecture for creating notebook cells with specific
functionality in the RDR system. Each cell type encapsulates its own logic, metadata,
and rendering behavior.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod


class CellType(Enum):
    """Enumeration of Jupyter cell types."""
    CODE = "code"
    MARKDOWN = "markdown"


@dataclass
class CellMetadata:
    """Metadata configuration for a notebook cell."""
    editable: bool = True
    deletable: bool = True
    init_cell: bool = False
    tags: List[str] = field(default_factory=list)
    jupyter: Dict[str, Any] = field(default_factory=dict)
    trusted: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary format for nbformat."""
        result = {
            'editable': self.editable,
            'deletable': self.deletable,
            'init_cell': self.init_cell,
            'tags': self.tags,
            'trusted': self.trusted
        }
        if self.jupyter:
            result['jupyter'] = self.jupyter
        return result


@dataclass
class NotebookCell(ABC):
    """
    Abstract base class for notebook cells.

    Each cell type defines its own content, metadata, and rendering behavior.
    """
    cell_type: CellType
    metadata: CellMetadata
    execution_count: Optional[int] = None
    outputs: List[Dict] = field(default_factory=list)

    @abstractmethod
    def get_source(self) -> str:
        """
        Get the source code/content for this cell.

        Returns:
            String containing the cell content.
        """
        pass

    def to_nbformat_dict(self) -> Dict[str, Any]:
        """
        Convert cell to nbformat dictionary structure.

        Returns:
            Dictionary compatible with nbformat.v4.new_code_cell or new_markdown_cell.
        """
        result = {
            'cell_type': self.cell_type.value,
            'source': self.get_source(),
            'metadata': self.metadata.to_dict()
        }

        if self.cell_type == CellType.CODE:
            result['execution_count'] = self.execution_count
            result['outputs'] = self.outputs

        return result


@dataclass
class InitializationCell(NotebookCell):
    """
    Cell that initializes notebook environment variables.

    Sets up communication file path, function name, and case name for the notebook.
    This cell is hidden and auto-executed on notebook load.
    """

    def __init__(self, comm_file: str, func_name: str, case_name: str,
                 connection_string: str = "rdr@localhost:3306/RDR"):
        self.comm_file = comm_file
        self.func_name = func_name
        self.case_name = case_name
        self.connection_string = connection_string

        metadata = CellMetadata(
            editable=False,
            deletable=False,
            init_cell=True,
            tags=['hide-input', 'hide-output', 'init_cell'],
            jupyter={'source_hidden': True, 'outputs_hidden': True}
        )

        super().__init__(
            cell_type=CellType.CODE,
            metadata=metadata
        )

    comm_file: str = ""
    func_name: str = ""
    case_name: str = ""
    connection_string: str = "rdr@localhost:3306/RDR"

    def __post_init__(self):
        if not hasattr(self, 'cell_type'):
            self.cell_type = CellType.CODE
        if not hasattr(self, 'metadata'):
            self.metadata = CellMetadata(
                editable=False,
                deletable=False,
                init_cell=True,
                tags=['hide-input', 'hide-output', 'init_cell'],
                jupyter={'source_hidden': True, 'outputs_hidden': True}
            )

    def get_source(self) -> str:
        return f"""import importlib
import sqlalchemy
from sqlalchemy import select
from ormatic.dao import *
from ripple_down_rules.orm_interface import *

# Configuration
COMM_FILE = r"{self.comm_file}"
TARGET_FUNC_NAME = "{self.func_name}"
case_name = "{self.case_name}"
connection_string = "{self.connection_string}"

# Database setup
engine = sqlalchemy.create_engine("mysql+pymysql://" + connection_string)
session = sqlalchemy.orm.Session(engine)
"""


@dataclass
class DatabaseQueryCell(NotebookCell):
    """
    Cell that queries the case from the database and reconstructs it.

    Uses SQLAlchemy to fetch the case object from MySQL and converts it
    from DAO format to the original case type.
    """
    case_name: str = ""

    def __init__(self, case_name: str):
        self.case_name = case_name

        metadata = CellMetadata(
            editable=False,
            deletable=False,
            init_cell=True,
            tags=['hide-input', 'init_cell'],
            jupyter={'source_hidden': True}
        )

        super().__init__(
            cell_type=CellType.CODE,
            metadata=metadata
        )


    def __post_init__(self):
        if not hasattr(self, 'cell_type'):
            self.cell_type = CellType.CODE
        if not hasattr(self, 'metadata'):
            self.metadata = CellMetadata(
                editable=False,
                deletable=False,
                init_cell=True,
                tags=['hide-input', 'init_cell'],
                jupyter={'source_hidden': True}
            )

    def get_source(self) -> str:
        return f"""# Query case from database
module_name = ".".join(case_name.split(".")[:-1])
case_type_name = case_name.split(".")[-1]
module = importlib.import_module(module_name)

# Fetch and reconstruct case
queried = session.scalar(select(get_dao_class(getattr(module, case_type_name))))
case = queried.from_dao() if queried else None

if case is None:
    print(f"Warning: Could not load case of type {{case_name}} from database")
"""


@dataclass
class VisualizationCell(NotebookCell):
    """
    Cell that displays SVG visualization of the rule tree.

    Creates an interactive display using ipywidgets to show object diagrams
    and rule tree visualizations. The case diagram is dynamic and updates
    when the Apply Rule button is clicked. The rule tree is static and shows
    the current state of the RDR knowledge base.
    """
    svg_data: Optional[str] = None

    def __init__(self, svg_data: Optional[str] = None):
        self.svg_data = svg_data

        metadata = CellMetadata(
            editable=False,
            deletable=False,
            init_cell=True,
            tags=['init_cell'],
            jupyter={'source_hidden': True}
        )

        super().__init__(
            cell_type=CellType.CODE,
            metadata=metadata
        )

    def __post_init__(self):
        if not hasattr(self, 'cell_type'):
            self.cell_type = CellType.CODE
        if not hasattr(self, 'metadata'):
            self.metadata = CellMetadata(
                editable=False,
                deletable=False,
                init_cell=True,
                tags=['init_cell'],
                jupyter={'source_hidden': True}
            )

    def get_source(self) -> str:
        base_code = """import ipywidgets as widgets
from IPython.display import display, Image, HTML
from ripple_down_rules.user_interface.object_diagram import generate_object_graph
import os

svg_output = widgets.HTML()

def render_svg(obj, name="Example"):
    \"\"\"Generate and display object diagram for the given object.\"\"\"
    try:
        graph = generate_object_graph(obj, name=name)
        svg_data = graph.pipe(format="svg").decode("utf-8")
        svg_output.value = svg_data
    except Exception as e:
        svg_output.value = f"<p style='color:red;'>Error generating diagram: {e}</p>"

# Display case object diagram (dynamic - updates when Apply Rule is clicked)
if case is not None:
    render_svg(case, name=case_name)
    display(widgets.HTML("<h3>Case Object Diagram</h3>"))
    display(svg_output)
else:
    display(widgets.HTML("<p style='color:orange;'>No case loaded</p>"))

# Display rule tree visualization (static - shows RDR knowledge base structure)
rule_tree_path = './rule_tree.svg'
if os.path.exists(rule_tree_path):
    display(widgets.HTML("<h3>Rule Tree</h3>"))
    try:
        display(Image(filename=rule_tree_path))
    except Exception as e:
        print(f"Could not display rule tree: {e}")
else:
    print("Rule tree visualization not available (no rules yet)")
"""
        return base_code


@dataclass
class FunctionCell(NotebookCell):
    """
    Cell containing the editable Python function for the user to modify.

    This is the main cell where users write their rule logic. It's the only
    cell that should be editable and visible by default.
    """
    boilerplate_code: str = ""

    def __init__(self, boilerplate_code: str):
        self.boilerplate_code = boilerplate_code

        metadata = CellMetadata(
            editable=True,
            deletable=False,
            init_cell=False,
            tags=[]
        )

        super().__init__(
            cell_type=CellType.CODE,
            metadata=metadata
        )

    def __post_init__(self):
        if not hasattr(self, 'cell_type'):
            self.cell_type = CellType.CODE
        if not hasattr(self, 'metadata'):
            self.metadata = CellMetadata(
                editable=True,
                deletable=False,
                init_cell=False,
                tags=[]
            )

    def get_source(self) -> str:
        return self.boilerplate_code


@dataclass
class SubmitButtonCell(NotebookCell):
    """
    Cell that creates the interactive toolbar with Apply and Accept buttons.

    - Apply button: Executes the function on the case and updates the visualization
    - Accept button: Saves the function source to the communication file and closes the notebook
    """
    func_name: str = ""

    def __init__(self, func_name: str):
        self.func_name = func_name

        metadata = CellMetadata(
            editable=False,
            deletable=False,
            init_cell=True,
            tags=['init_cell'],
            jupyter={'source_hidden': True}
        )

        super().__init__(
            cell_type=CellType.CODE,
            metadata=metadata
        )

    def __post_init__(self):
        if not hasattr(self, 'cell_type'):
            self.cell_type = CellType.CODE
        if not hasattr(self, 'metadata'):
            self.metadata = CellMetadata(
                editable=False,
                deletable=False,
                init_cell=True,
                tags=['init_cell'],
                jupyter={'source_hidden': True}
            )

    def get_source(self) -> str:
        return f"""import os
import time
import json
import re
from IPython import get_ipython
import ipywidgets as widgets
from IPython.display import display

# Create toolbar buttons
apply_rule_btn = widgets.Button(description="Apply Rule", button_style="success")
accept_rule_btn = widgets.Button(description="Accept Rule", button_style="primary")
toolbar = widgets.HBox([apply_rule_btn, accept_rule_btn])
toolbar.add_class('rdr-toolbar')

def on_apply_rule(btn):
    \"\"\"Apply the function to the case and refresh visualization.\"\"\"
    try:
        fn = globals().get(TARGET_FUNC_NAME)
        if callable(fn) and case is not None:
            result = fn(case)
            print(f"Function returned: {{result}}")
            render_svg(case, name=case_name)
        elif not callable(fn):
            print(f"Function '{{TARGET_FUNC_NAME}}' not found in namespace")
        else:
            print("No case loaded to apply function to")
    except Exception as e:
        print(f"Error applying function: {{e}}")
        import traceback
        traceback.print_exc()

def send_cells_to_main_process(btn):
    \"\"\"Extract function source and write to communication file.\"\"\"
    print("Sending function back to main process...")
    try:
        ns = get_ipython().user_ns
        cells = ns.get('_ih') or ns.get('In') or []
        target = None
        pattern = re.compile(r'^\\s*(async\\s+)?def\\s+' + re.escape(TARGET_FUNC_NAME) + r'\\s*\\(', flags=re.M)
        
        for content in reversed(cells):
            if isinstance(content, str) and pattern.search(content):
                target = content
                break
        
        if target:
            with open(COMM_FILE, 'w') as f:
                json.dump({{'timestamp': time.time(), 'function_source': target}}, f)
                f.flush()
                os.fsync(f.fileno())
            print(f"Function source written to: {{COMM_FILE}}")
        else:
            print(f"Function '{{TARGET_FUNC_NAME}}' not found in notebook history")
    except Exception as e:
        print(f"Error writing function: {{e}}")
        import traceback
        traceback.print_exc()
    
    # Best-effort visualization update
    try:
        on_apply_rule(btn)
    except Exception:
        pass

apply_rule_btn.on_click(on_apply_rule)
accept_rule_btn.on_click(send_cells_to_main_process)

display(toolbar)
"""


def create_standard_notebook_cells(
    case_name: str,
    func_name: str,
    boilerplate_code: str,
    comm_file: str,
    connection_string: str = "rdr@localhost:3306/RDR",
    svg_data: Optional[str] = None
) -> List[NotebookCell]:
    """
    Create the standard set of cells for an RDR notebook.

    Args:
        case_name: Full qualified class name of the case (e.g., "ripple_down_rules.datasets.Robot")
        func_name: Name of the function to be edited
        boilerplate_code: Initial function code for the user to modify
        comm_file: Path to the JSON communication file
        connection_string: Database connection string
        svg_data: Optional SVG data for rule tree visualization

    Returns:
        List of NotebookCell instances in the correct order.
    """
    cells = [
        InitializationCell(
            comm_file=comm_file,
            func_name=func_name,
            case_name=case_name,
            connection_string=connection_string
        ),
        DatabaseQueryCell(case_name=case_name),
        VisualizationCell(svg_data=svg_data),
        FunctionCell(boilerplate_code=boilerplate_code),
        SubmitButtonCell(func_name=func_name)
    ]
    return cells