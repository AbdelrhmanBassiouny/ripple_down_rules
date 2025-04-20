from __future__ import annotations

import ast
import importlib
import json
import logging
import os
from abc import abstractmethod
from collections import UserDict
from copy import deepcopy
from dataclasses import dataclass

import matplotlib
import networkx as nx
from anytree import Node, RenderTree
from anytree.exporter import DotExporter
from matplotlib import pyplot as plt
from sqlalchemy import MetaData, inspect
from sqlalchemy.orm import Mapped, registry, class_mapper, DeclarativeBase as SQLTable, Session
from tabulate import tabulate
from typing_extensions import Callable, Set, Any, Type, Dict, TYPE_CHECKING, get_type_hints, \
    get_origin, get_args, Tuple, Optional, List, Union, Self

if TYPE_CHECKING:
    from .datastructures import Case

matplotlib.use("Qt5Agg")  # or "Qt5Agg", depending on availability


def typing_to_python_type(typing_hint: Type) -> Type:
    """
    Convert a typing hint to a python type.

    :param typing_hint: The typing hint to convert.
    :return: The python type.
    """
    if typing_hint in [list, List]:
        return list
    elif typing_hint in [tuple, Tuple]:
        return tuple
    elif typing_hint in [set, Set]:
        return set
    elif typing_hint in [dict, Dict]:
        return dict
    else:
        return typing_hint


def capture_variable_assignment(code: str, variable_name: str) -> Optional[str]:
    """
    Capture the assignment of a variable in the given code.

    :param code: The code to analyze.
    :param variable_name: The name of the variable to capture.
    :return: The assignment statement or None if not found.
    """
    tree = ast.parse(code)
    assignment = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == variable_name:
                    # now extract the right side of the assignment
                    assignment = ast.get_source_segment(code, node.value)
                    break
        if assignment is not None:
            break
    return assignment


def get_func_rdr_model_path(func: Callable, model_dir: str) -> str:
    """
    :param func: The function to get the model path for.
    :param model_dir: The directory to save the model to.
    :return: The path to the model file.
    """
    func_name = get_method_name(func)
    func_class_name = get_method_class_name_if_exists(func)
    func_file_name = get_method_file_name(func)
    model_name = func_file_name
    model_name += f"_{func_class_name}" if func_class_name else ""
    model_name += f"_{func_name}"
    return os.path.join(model_dir, f"{model_name}.json")


def get_method_args_as_dict(method: Callable, *args, **kwargs) -> Dict[str, Any]:
    """
    Get the arguments of a method as a dictionary.

    :param method: The method to get the arguments from.
    :param args: The positional arguments.
    :param kwargs: The keyword arguments.
    :return: A dictionary of the arguments.
    """
    func_arg_names = method.__code__.co_varnames
    func_arg_values = args + tuple(kwargs.values())
    return dict(zip(func_arg_names, func_arg_values))


def get_method_name(method: Callable) -> str:
    """
    Get the name of a method.

    :param method: The method to get the name of.
    :return: The name of the method.
    """
    return method.__name__ if hasattr(method, "__name__") else str(method)


def get_method_class_name_if_exists(method: Callable) -> Optional[str]:
    """
    Get the class name of a method if it has one.

    :param method: The method to get the class name of.
    :return: The class name of the method.
    """
    if hasattr(method, "__self__") and hasattr(method.__self__, "__class__"):
        return method.__self__.__class__.__name__
    return None


def get_method_file_name(method: Callable) -> str:
    """
    Get the file name of a method.

    :param method: The method to get the file name of.
    :return: The file name of the method.
    """
    return method.__code__.co_filename


def flatten_list(a: List):
    a_flattened = []
    for c in a:
        if is_iterable(c):
            a_flattened.extend(list(c))
        else:
            a_flattened.append(c)
    return a_flattened


def make_list(value: Any) -> List:
    """
    Make a list from a value.

    :param value: The value to make a list from.
    """
    return list(value) if is_iterable(value) else [value]


def is_iterable(obj: Any) -> bool:
    """
    Check if an object is iterable.

    :param obj: The object to check.
    """
    return hasattr(obj, "__iter__") and not isinstance(obj, (str, type))


def get_type_from_string(type_path: str):
    """
    Get a type from a string describing its path using the format "module_path.ClassName".

    :param type_path: The path to the type.
    """
    module_path, class_name = type_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_full_class_name(cls):
    """
    Returns the full name of a class, including the module name.
    Copied from: https://github.com/tomsch420/random-events/blob/master/src/random_events/utils.py#L6C1-L21C101

    :param cls: The class.
    :return: The full name of the class
    """
    return cls.__module__ + "." + cls.__name__


def recursive_subclasses(cls):
    """
    Copied from: https://github.com/tomsch420/random-events/blob/master/src/random_events/utils.py#L6C1-L21C101
    :param cls: The class.
    :return: A list of the classes subclasses.
    """
    return cls.__subclasses__() + [g for s in cls.__subclasses__() for g in recursive_subclasses(s)]


class SubclassJSONSerializer:
    """
    Originally from: https://github.com/tomsch420/random-events/blob/master/src/random_events/utils.py#L6C1-L21C101
    Class for automatic (de)serialization of subclasses.
    Classes that inherit from this class can be serialized and deserialized automatically by calling this classes
    'from_json' method.
    """

    def to_json_file(self, filename: str):
        """
        Save the object to a json file.
        """
        data = self.to_json()
        # save the json to a file
        if not filename.endswith(".json"):
            filename += ".json"
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
        return data

    def to_json(self) -> Dict[str, Any]:
        return {"_type": get_full_class_name(self.__class__), **self._to_json()}

    def _to_json(self) -> Dict[str, Any]:
        """
        Create a json dict from the object.
        """
        raise NotImplementedError()

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        """
        Create a variable from a json dict.
        This method is called from the from_json method after the correct subclass is determined and should be
        overwritten by the respective subclass.

        :param data: The json dict
        :return: The deserialized object
        """
        raise NotImplementedError()

    @classmethod
    def from_json_file(cls, filename: str):
        """
        Create an instance of the subclass from the data in the given json file.

        :param filename: The filename of the json file.
        """
        if not filename.endswith(".json"):
            filename += ".json"
        with open(filename, "r") as f:
            scrdr_json = json.load(f)
        return cls.from_json(scrdr_json)

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> Self:
        """
        Create the correct instanceof the subclass from a json dict.

        :param data: The json dict
        :return: The correct instance of the subclass
        """
        if data is None:
            return None
        if not isinstance(data, dict) or ('_type' not in data):
            return data
        # check if type module is builtins
        data_type = get_type_from_string(data["_type"])
        if data_type.__module__ == 'builtins':
            if is_iterable(data['value']) and not isinstance(data['value'], dict):
                return data_type([cls.from_json(d) for d in data['value']])
            return data_type(data["value"])
        if get_full_class_name(cls) == data["_type"]:
            data.pop("_type")
            return cls._from_json(data)
        for subclass in recursive_subclasses(SubclassJSONSerializer):
            if get_full_class_name(subclass) == data["_type"]:
                subclass_data = deepcopy(data)
                subclass_data.pop("_type")
                return subclass._from_json(subclass_data)

        raise ValueError("Unknown type {}".format(data["_type"]))

    save = to_json_file
    load = from_json_file


def copy_case(case: Union[Case, SQLTable]) -> Union[Case, SQLTable]:
    """
    Copy a case.

    :param case: The case to copy.
    :return: The copied case.
    """
    if isinstance(case, SQLTable):
        return copy_orm_instance_with_relationships(case)
    else:
        return deepcopy(case)


def copy_orm_instance(instance: SQLTable) -> SQLTable:
    """
    Copy an ORM instance by expunging it from the session then deep copying it and adding it back to the session. This
    is useful when you want to copy an instance and make changes to it without affecting the original instance.

    :param instance: The instance to copy.
    :return: The copied instance.
    """
    session: Session = inspect(instance).session
    session.expunge(instance)
    new_instance = deepcopy(instance)
    session.add(instance)
    return new_instance


def copy_orm_instance_with_relationships(instance: SQLTable) -> SQLTable:
    """
    Copy an ORM instance with its relationships (i.e. its foreign keys).

    :param instance: The instance to copy.
    :return: The copied instance.
    """
    instance_cp = copy_orm_instance(instance)
    for rel in class_mapper(instance.__class__).relationships:
        related_obj = getattr(instance, rel.key)
        if related_obj is not None:
            setattr(instance_cp, rel.key, related_obj)
    return instance_cp


def get_value_type_from_type_hint(attr_name: str, obj: Any) -> Type:
    """
    Get the value type from the type hint of an object attribute.

    :param attr_name: The name of the attribute.
    :param obj: The object to get the attributes from.
    """
    hint, origin, args = get_hint_for_attribute(attr_name, obj)
    if not origin and not hint:
        if hasattr(obj, attr_name):
            attr_value = getattr(obj, attr_name)
            if attr_value is not None:
                return type(attr_value)
        raise ValueError(f"Couldn't get type for Attribute {attr_name}, please provide a type hint")
    if origin in [list, set, tuple, type, dict]:
        attr_value_type = args[0]
    elif hint:
        attr_value_type = hint
    else:
        raise ValueError(f"Attribute {attr_name} has unsupported type {hint}.")
    return attr_value_type


def get_hint_for_attribute(attr_name: str, obj: Any) -> Tuple[Optional[Any], Optional[Any], Tuple[Any]]:
    """
    Get the type hint for an attribute of an object.

    :param attr_name: The name of the attribute.
    :param obj: The object to get the attribute from.
    :return: The type hint of the attribute.
    """
    if attr_name is None or not hasattr(obj.__class__, attr_name):
        return None, None, ()
    class_attr = getattr(obj.__class__, attr_name)
    if isinstance(class_attr, property):
        if not class_attr.fget:
            raise ValueError(f"Attribute {attr_name} has no getter.")
        hint = get_type_hints(class_attr.fget)['return']
    else:
        try:
            hint = get_type_hints(obj.__class__)[attr_name]
        except KeyError:
            hint = type(class_attr)
    origin = get_origin(hint)
    args = get_args(hint)
    if origin is Mapped:
        return args[0], get_origin(args[0]), get_args(args[0])
    else:
        return hint, origin, args


def table_rows_as_str(row_dict: Dict[str, Any], columns_per_row: int = 9):
    """
    Print a table row.

    :param row_dict: The row to print.
    :param columns_per_row: The maximum number of columns per row.
    """
    all_items = list(row_dict.items())
    # make items a list of n rows such that each row has a max size of 4
    all_items = [all_items[i:i + columns_per_row] for i in range(0, len(all_items), columns_per_row)]
    keys = [list(map(lambda i: i[0], row)) for row in all_items]
    values = [list(map(lambda i: i[1], row)) for row in all_items]
    all_table_rows = []
    for row_keys, row_values in zip(keys, values):
        table = tabulate([row_values], headers=row_keys, tablefmt='plain', maxcolwidths=[20] * len(row_keys))
        all_table_rows.append(table)
    return "\n".join(all_table_rows)


def row_to_dict(obj):
    return {
        col.name: getattr(obj, col.name)
        for col in obj.__table__.columns
        if not col.primary_key and not col.foreign_keys
    }


def get_attribute_name(obj: Any, attribute: Optional[Any] = None, attribute_type: Optional[Type] = None,
                       possible_value: Optional[Any] = None) -> Optional[str]:
    """
    Get the name of an attribute from an object. The attribute can be given as a value, a type or a target value.
    And this method will try to find the attribute name using the given information.

    :param obj: The object to get the attribute name from.
    :param attribute: The attribute to get the name of.
    :param attribute_type: The type of the attribute to get the name of.
    :param possible_value: A possible value of the attribute to get the name of.
    :return: The name of the attribute.
    """
    attribute_name: Optional[str] = None
    if attribute_name is None and attribute is not None:
        attribute_name = get_attribute_name_from_value(obj, attribute)
    if attribute_name is None and attribute_type is not None:
        attribute_name = get_attribute_by_type(obj, attribute_type)[0]
    if attribute_name is None and possible_value is not None:
        attribute_name = get_attribute_by_type(obj, type(possible_value))[0]
    return attribute_name


def get_attribute_by_type(obj: Any, prop_type: Type) -> Tuple[Optional[str], Optional[Any]]:
    """
    Get a property from an object by type.

    :param obj: The object to get the property from.
    :param prop_type: The type of the property.
    """
    for name in dir(obj):
        if name.startswith("_") or callable(getattr(obj, name)):
            continue
        if isinstance(getattr(obj, name), (MetaData, registry)):
            continue
        prop_value = getattr(obj, name)
        if isinstance(prop_value, prop_type):
            return name, prop_value
        if hasattr(prop_value, "__iter__") and not isinstance(prop_value, str):
            if len(prop_value) > 0 and any(isinstance(v, prop_type) for v in prop_value):
                return name, prop_value
            else:
                # get args of type hint
                hint, origin, args = get_hint_for_attribute(name, obj)
                if origin in [list, set, tuple, dict, List, Set, Tuple, Dict]:
                    if prop_type is args[0]:
                        return name, prop_value
        else:
            # get the type hint of the attribute
            hint, origin, args = get_hint_for_attribute(name, obj)
            if hint is prop_type:
                return name, prop_value
            elif origin in [list, set, tuple, dict, List, Set, Tuple, Dict]:
                if prop_type is args[0]:
                    return name, prop_value
    return None, None


def get_attribute_name_from_value(obj: Any, attribute_value: Any) -> Optional[str]:
    """
    Get the name of an attribute from an object.

    :param obj: The object to get the attribute name from.
    :param attribute_value: The attribute value to get the name of.
    """
    for name in dir(obj):
        if name.startswith("_") or callable(getattr(obj, name)):
            continue
        prop_value = getattr(obj, name)
        if prop_value is attribute_value:
            return name


def get_attribute_values_transitively(obj: Any, attribute: Any) -> Any:
    """
    Get an attribute from a python object, if it is iterable, get the attribute values from all elements and unpack them
    into a list.

    :param obj: The object to get the sub attribute from.
    :param attribute: The  attribute to get.
    """
    if hasattr(obj, "__iter__") and not isinstance(obj, str):
        if isinstance(obj, (dict, UserDict)):
            all_values = [get_attribute_values_transitively(v, attribute) for v in obj.values()
                          if not isinstance(v, (str, type)) and hasattr(v, attribute)]
        else:
            all_values = [get_attribute_values_transitively(a, attribute) for a in obj
                          if not isinstance(a, (str, type)) and hasattr(a, attribute)]
        if can_be_a_set(all_values):
            return set().union(*all_values)
        else:
            return set(all_values)
    return getattr(obj, attribute)


def can_be_a_set(value: Any) -> bool:
    """
    Check if a value can be a set.

    :param value: The value to check.
    """
    if hasattr(value, "__iter__") and not isinstance(value, str):
        if len(value) > 0 and any(hasattr(v, "__iter__") and not isinstance(v, str) for v in value):
            return False
        else:
            return True
    else:
        return False


def get_all_subclasses(cls: Type) -> Dict[str, Type]:
    """
    Get all subclasses of a class recursively.

    :param cls: The class to get the subclasses of.
    :return: A dictionary of all subclasses.
    """
    all_subclasses: Dict[str, Type] = {}
    for sub_cls in cls.__subclasses__():
        all_subclasses[sub_cls.__name__.lower()] = sub_cls
        all_subclasses.update(get_all_subclasses(sub_cls))
    return all_subclasses


def make_set(value: Any) -> Set:
    """
    Make a set from a value.

    :param value: The value to make a set from.
    """
    return set(value) if is_iterable(value) else {value}


def make_value_or_raise_error(value: Any) -> Any:
    """
    Make a value or raise an error if the value is not a single value.

    :param value: The value to check.
    """
    if hasattr(value, "__iter__") and not isinstance(value, str):
        if hasattr(value, "__len__") and len(value) == 1:
            return list(value)[0]
        else:
            raise ValueError(f"Expected a single value, got {value}")
    return value


def tree_to_graph(root_node: Node) -> nx.DiGraph:
    """
    Convert anytree to a networkx graph.

    :param root_node: The root node of the tree.
    :return: A networkx graph.
    """
    graph = nx.DiGraph()
    unique_node_names = get_unique_node_names_func(root_node)

    def add_edges(node):
        if unique_node_names(node) not in graph.nodes:
            graph.add_node(unique_node_names(node))
        for child in node.children:
            if unique_node_names(child) not in graph.nodes:
                graph.add_node(unique_node_names(child))
            graph.add_edge(unique_node_names(node), unique_node_names(child), weight=child.weight)
            add_edges(child)

    add_edges(root_node)
    return graph


def get_unique_node_names_func(root_node) -> Callable[[Node], str]:
    nodes = [root_node]

    def get_all_nodes(node):
        for c in node.children:
            nodes.append(c)
            get_all_nodes(c)

    get_all_nodes(root_node)

    def nodenamefunc(node: Node):
        """
        Set the node name for the dot exporter.
        """
        similar_nodes = [n for n in nodes if n.name == node.name]
        node_idx = similar_nodes.index(node)
        return node.name if node_idx == 0 else f"{node.name}_{node_idx}"

    return nodenamefunc


def edge_attr_setter(parent, child):
    """
    Set the edge attributes for the dot exporter.
    """
    if child and hasattr(child, "weight") and child.weight:
        return f'style="bold", label=" {child.weight}"'
    return ""


def render_tree(root: Node, use_dot_exporter: bool = False,
                filename: str = "scrdr"):
    """
    Render the tree using the console and optionally export it to a dot file.

    :param root: The root node of the tree.
    :param use_dot_exporter: Whether to export the tree to a dot file.
    :param filename: The name of the file to export the tree to.
    """
    if not root:
        logging.warning("No rules to render")
        return
    for pre, _, node in RenderTree(root):
        print(f"{pre}{node.weight if hasattr(node, 'weight') and node.weight else ''} {node.__str__()}")
    if use_dot_exporter:
        unique_node_names = get_unique_node_names_func(root)

        de = DotExporter(root,
                         nodenamefunc=unique_node_names,
                         edgeattrfunc=edge_attr_setter
                         )
        de.to_dotfile(f"{filename}{'.dot'}")
        de.to_picture(f"{filename}{'.png'}")


def draw_tree(root: Node, fig: plt.Figure):
    """
    Draw the tree using matplotlib and networkx.
    """
    if root is None:
        return
    fig.clf()
    graph = tree_to_graph(root)
    fig_sz_x = 13
    fig_sz_y = 10
    fig.set_size_inches(fig_sz_x, fig_sz_y)
    pos = nx.drawing.nx_agraph.graphviz_layout(graph, prog="dot")
    # scale down pos
    max_pos_x = max([v[0] for v in pos.values()])
    max_pos_y = max([v[1] for v in pos.values()])
    pos = {k: (v[0] * fig_sz_x / max_pos_x, v[1] * fig_sz_y / max_pos_y) for k, v in pos.items()}
    nx.draw(graph, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=1000,
            ax=fig.gca(), node_shape="o", font_size=8)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=nx.get_edge_attributes(graph, 'weight'),
                                 ax=fig.gca(), rotate=False, clip_on=False)
    plt.pause(0.1)
