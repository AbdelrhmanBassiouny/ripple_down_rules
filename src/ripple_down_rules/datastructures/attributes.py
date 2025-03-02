from collections import UserDict

from typing_extensions import Any, Optional, Dict

from ripple_down_rules.datastructures import Attribute, ValueType, Categorical, get_or_create_matching_attribute, \
    CustomSet, get_value_type_from_type_hint
from ripple_down_rules.utils import make_set, get_attribute_values_transitively, can_be_a_set


class Attributes(UserDict):
    """
    A collection of attributes that represents a set of constraints on a case. This is a dictionary where the keys are
    the names of the attributes and the values are the attributes. All are stored in lower case.
    """

    def __getitem__(self, item: str) -> Any:
        return super().__getitem__(item.lower())

    def __setitem__(self, name: str, value: Any):
        name = name.lower()
        if name in self:
            if isinstance(self[name], set):
                value = make_set(value)
                current_element_class = list(self[name])[0].__class__
                element_class = list(self[name])[0].__class__
                if len(self[name]) > 0 and not issubclass(current_element_class, element_class):
                    raise ValueError(f"Attribute {name} already exists in the case and is mutually exclusive.")
                else:
                    self[name].update(make_set(value))
            elif not issubclass(self[name].__class__, value.__class__):
                raise ValueError(f"Attribute {name} already exists in the case and is of a different type,"
                                 f"current: {self[name].__class__} != new: {value.__class__}")
            else:
                raise ValueError(f"Attribute {name} already exists in the case and is mutually exclusive.")
        else:
            super().__setitem__(name, value)

    def __contains__(self, item):
        return super().__contains__(item.lower())

    def __delitem__(self, key):
        super().__delitem__(key.lower())

    def __eq__(self, other):
        if not isinstance(other, (Attributes, dict)):
            return False
        elif isinstance(other, dict):
            return super().__eq__(Attributes(other))
        return super().__eq__(other)


def get_all_possible_contexts(obj: Any, recursion_idx: int = 0, max_recursion_idx: int = 1,
                              start_with_name: Optional[str] = None, parent_iterable: bool = False) -> Dict[str, Any]:
    """
    Get all possible contexts for an object.

    :param obj: The object to get the contexts for.
    :param recursion_idx: The recursion index to prevent infinite recursion.
    :param max_recursion_idx: The maximum recursion index.
    :param start_with_name: The starting context.
    :param parent_iterable: Whether the parent is an iterable.
    :return: A dictionary of all possible contexts.
    """
    all_contexts = Attributes()
    if recursion_idx > max_recursion_idx:
        return all_contexts
    for attr in dir(obj):
        if attr.startswith("__") or attr.startswith("_") or callable(getattr(obj, attr)):
            continue
        # attr_value = get_attribute_values_transitively(obj, attr)
        attr_value = getattr(obj, attr)
        chained_name = f"{start_with_name}.{attr}" if start_with_name else attr
        if isinstance(attr_value, (dict, UserDict)):
            all_contexts.update({f"{chained_name}.{k}": v for k, v in attr_value.items()})
        if hasattr(attr_value, "__iter__") and not isinstance(attr_value, str):
            values = attr_value.values() if isinstance(attr_value, (dict, UserDict)) else attr_value
            all_vals_contexts = Attributes()
            for idx, val in enumerate(values):
                if hasattr(val, "__iter__"):
                    continue
                val_context = get_all_possible_contexts(val, recursion_idx=recursion_idx + 1,
                                                        max_recursion_idx=max_recursion_idx,
                                                        start_with_name=chained_name,
                                                        parent_iterable=True)
                all_vals_contexts.update(val_context)
            range_ = {type(list(values)[0])} if len(values) > 0 else set()
            if len(range_) == 0:
                range_ = make_set(get_value_type_from_type_hint(attr, obj))
            new_attr_type = CustomSet.create(attr, range_)
            new_attr = new_attr_type(make_set(values))
            for name, val in all_vals_contexts.items():
                setattr(new_attr, name.replace(f"{chained_name}.", ""), val)
            all_contexts[chained_name] = new_attr
            all_contexts.update(all_vals_contexts)
        else:
            sub_attr_contexts = get_all_possible_contexts(getattr(obj, attr), recursion_idx=recursion_idx + 1,
                                                          max_recursion_idx=max_recursion_idx,
                                                          start_with_name=chained_name)
            all_contexts.update(sub_attr_contexts)
            all_contexts[chained_name] = make_set(attr_value) \
                if can_be_a_set(attr_value) or parent_iterable else attr_value
    return all_contexts
