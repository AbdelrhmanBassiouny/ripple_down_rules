from __future__ import annotations

import math

import numpy as np
from tf.transformations import quaternion_inverse, quaternion_multiply
from typing_extensions import List, Optional, Callable, Tuple

import pycrap
from pycram.datastructures.dataclasses import (ContactPointsList, AxisAlignedBoundingBox as AABB)
from pycram.datastructures.pose import Transform
from pycram.datastructures.world import World, UseProspectionWorld
from pycram.datastructures.world_entity import PhysicalBody
from pycram.object_descriptors.generic import ObjectDescription as GenericObjectDescription
from pycram.world_concepts.world_object import Object


def str_to_operator_fn(rule_str: str) -> Tuple[Optional[str], Optional[str], Optional[Callable]]:
    """
    Convert a string containing a rule to a function that represents the rule.

    :param rule_str: A string that contains the rule.
    :return: A function that represents the rule.
    """
    func = None
    op = None
    arg1 = None
    arg2 = None
    if '=' not in rule_str:
        if "<" in rule_str:
            func = lambda x, y: x < y
            op = "<"
        elif ">" in rule_str:
            func = lambda x, y: x > y
            op = ">"
    elif '=' in rule_str:
        if "<=" in rule_str:
            func = lambda x, y: x <= y
            op = "<="
        elif ">=" in rule_str:
            func = lambda x, y: x >= y
            op = ">="
        elif "==" in rule_str:
            func = lambda x, y: x == y
            op = "=="
    if op is not None:
        arg1, arg2 = rule_str.split(op)
        arg1 = arg1.strip()
        arg2 = arg2.strip()
    return arg1, arg2, func


def check_if_object_is_supported(obj: Object, distance: Optional[float] = 0.03) -> bool:
    """
    Check if the object is supported by any other object.

    :param obj: The object to check if it is supported.
    :param distance: The distance to check if the object is supported.
    :return: True if the object is supported, False otherwise.
    """
    supported = True
    with UseProspectionWorld():
        prospection_obj = World.current_world.get_prospection_object_for_object(obj)
        dt = math.sqrt(2 * distance / 9.81) + 0.01  # time to fall distance
        World.current_world.simulate(dt)
        cp = prospection_obj.contact_points
        if not check_if_in_contact_with_support(prospection_obj, cp.get_bodies_in_contact()):
            return False
    return supported


def check_if_object_is_supported_using_contact_points(obj: Object, contact_points: ContactPointsList) -> bool:
    """
    Check if the object is supported by any other object using the contact points.

    :param obj: The object to check if it is supported.
    :param contact_points: The contact points of the object.
    :return: True if the object is supported, False otherwise.
    """
    for body in contact_points.get_bodies_in_contact():
        if check_if_object_is_supported_by_another_object(obj, body, contact_points.get_points_of_body(body)):
            return True


def check_if_in_contact_with_support(obj: Object, contact_bodies: List[PhysicalBody]) -> Optional[PhysicalBody]:
    """
    Check if the object is in contact with a supporting surface.

    :param obj: The object to check if it is in contact with a supporting surface.
    :param contact_bodies: The bodies in contact with the object.
    """
    for body in contact_bodies:
        if issubclass(body.parent_entity.obj_type, (pycrap.Supporter, pycrap.Location)):
            body_aabb = body.get_axis_aligned_bounding_box()
            surface_z = body_aabb.max_z
            tracked_object_base = obj.get_base_origin().position
            if tracked_object_base.z >= surface_z and body_aabb.min_x <= tracked_object_base.x <= body_aabb.max_x and \
                    body_aabb.min_y <= tracked_object_base.y <= body_aabb.max_y:
                return body


def check_if_object_is_supported_by_another_object(obj: Object, support_obj: Object,
                                                   contact_points: Optional[ContactPointsList] = None) -> bool:
    """
    Check if the object is supported by another object.

    :param obj: The object to check if it is supported.
    :param support_obj: The object that supports the object.
    :param contact_points: The contact points between the object and the support object.
    :return: True if the object is supported by the support object, False otherwise.
    """
    if contact_points is None:
        contact_points = obj.get_contact_points_with_body(support_obj)
    normals = [cp.normal for cp in contact_points if any(cp.normal)]
    if len(normals) > 0:
        average_normal = np.mean(normals, axis=0)
        return is_vector_opposite_to_gravity(average_normal)
    return False


def is_vector_opposite_to_gravity(vector: List[float], gravity_vector: Optional[List[float]] = None) -> bool:
    """
    Check if the vector is opposite to the gravity vector.

    :param vector: A list of float values that represent the vector.
    :param gravity_vector: A list of float values that represent the gravity vector.
    :return: True if the vector is opposite to the gravity vector, False otherwise.
    """
    gravity_vector = [0, 0, -1] if gravity_vector is None else gravity_vector
    return np.dot(vector, gravity_vector) < 0


class Imaginator:
    """
    A class that provides methods for imagining objects.
    """
    surfaces_created: List[Object] = []
    latest_surface_idx: int = 0

    @classmethod
    def imagine_support_from_aabb(cls, aabb: AABB) -> Object:
        """
        Imagine a support with the size of the axis-aligned bounding box.

        :param aabb: The axis-aligned bounding box for which the support of same size should be imagined.
        :return: The support object.
        """
        return cls._imagine_support(aabb=aabb)

    @classmethod
    def imagine_support_for_object(cls, obj: Object, support_thickness: Optional[float] = 0.005) -> Object:
        """
        Imagine a support that supports the object and has a specified thickness.

        :param obj: The object for which the support should be imagined.
        :param support_thickness: The thickness of the support.
        :return: The support object
        """
        return cls._imagine_support(obj=obj, support_thickness=support_thickness)

    @classmethod
    def _imagine_support(cls, obj: Optional[Object] = None,
                         aabb: Optional[AABB] = None,
                         support_thickness: Optional[float] = None) -> Object:
        """
        Imagine a support for the object or with the size of the axis-aligned bounding box.

        :param obj: The object for which the support should be imagined.
        :param aabb: The axis-aligned bounding box for which the support of same size should be imagined.
        :param support_thickness: The thickness of the support.
        :return: The support object.
        """
        if aabb is not None:
            obj_aabb = aabb
        elif obj is not None:
            obj_aabb = obj.get_axis_aligned_bounding_box()
        else:
            raise ValueError("Either object or axis-aligned bounding box should be provided.")
        print(f"support index: {cls.latest_surface_idx}")
        support_name = f"imagined_support_{cls.latest_surface_idx}"
        support_thickness = obj_aabb.depth if support_thickness is None else support_thickness
        support = GenericObjectDescription(support_name,
                                           [0, 0, 0], [obj_aabb.width, obj_aabb.depth, support_thickness * 0.5])
        support_obj = Object(support_name, pycrap.Supporter, None, support)
        support_position = obj_aabb.base_origin
        support_obj.set_position(support_position)
        cp = support_obj.contact_points
        contacted_objects = cp.get_objects_that_have_points()
        contacted_surfaces = [obj for obj in contacted_objects if obj in cls.surfaces_created]
        for obj in contacted_surfaces:
            support_obj = support_obj.merge(obj)
            cls.surfaces_created.remove(obj)
        World.current_world.get_object_by_type(pycrap.Floor)[0].attach(support_obj)
        cls.surfaces_created.append(support_obj)
        cls.latest_surface_idx += 1
        return support_obj


def get_angle_between_vectors(vector_1: List[float], vector_2: List[float]) -> float:
    """
    Get the angle between two vectors.

    :param vector_1: A list of float values that represent the first vector.
    :param vector_2: A list of float values that represent the second vector.
    :return: A float value that represents the angle between the two vectors.
    """
    return np.arccos(np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2)))


def calculate_transform_difference_and_check_if_small(transform_1: Transform, transform_2: Transform,
                                                      translation_threshold: float, angle_threshold: float) -> bool:
    """
    Calculate the translation and rotation of the object with respect to the hand to check if it was picked up,
     uses the translation and rotation thresholds to determine if the object was picked up.

    :param transform_1: The transform of the object at the first time step.
    :param transform_2: The transform of the object at the second time step.
    :param translation_threshold: The threshold for the translation difference to be considered as small.
    :param angle_threshold: The threshold for the angle between the two quaternions to be considered as small.
    :return: A tuple of two boolean values that represent the conditions for the translation and rotation of the
    object to be considered as picked up.
    """
    trans_1, quat_1 = transform_1.translation_as_list(), transform_1.rotation_as_list()
    trans_2, quat_2 = transform_2.translation_as_list(), transform_2.rotation_as_list()
    trans_diff_cond = calculate_translation_difference_and_check(trans_1, trans_2, translation_threshold)
    rot_diff_cond = calculate_angle_between_quaternions_and_check(quat_1, quat_2, angle_threshold)
    return trans_diff_cond and rot_diff_cond


def calculate_translation_difference_and_check(trans_1: List[float], trans_2: List[float],
                                               threshold: float) -> bool:
    """
    Calculate the translation difference and checks if it is small.

    :param trans_1: The translation of the object at the first time step.
    :param trans_2: The translation of the object at the second time step.
    :param threshold: The threshold for the translation difference to be considered as small.
    :return: A boolean value that represents the condition for the translation of the object to be considered as
    picked up.
    """
    translation_diff = calculate_abs_translation_difference(trans_1, trans_2)
    return is_translation_difference_small(translation_diff, threshold)


def is_translation_difference_small(trans_diff: List[float], threshold: float) -> bool:
    """
    Check if the translation difference is small by comparing it to the translation threshold.

    :param trans_diff: The translation difference.
    :param threshold: The threshold for the translation difference to be considered as small.
    :return: A boolean value that represents the condition for the translation difference to be considered as small.
    """
    return np.linalg.norm(trans_diff) <= threshold
    # return all([diff <= threshold for diff in trans_diff])


def calculate_translation(position_1: List[float], position_2: List[float]) -> List:
    """
    calculate the translation between two positions.

    :param position_1: The first position.
    :param position_2: The second position.
    :return: A list of float values that represent the translation between the two positions.
    """
    return [p2 - p1 for p1, p2 in zip(position_1, position_2)]


def calculate_abs_translation_difference(trans_1: List[float], trans_2: List[float]) -> List[float]:
    """
    Calculate the translation difference.

    :param trans_1: The translation of the object at the first time step.
    :param trans_2: The translation of the object at the second time step.
    :return: A list of float values that represent the translation difference.
    """
    return [abs(t1 - t2) for t1, t2 in zip(trans_1, trans_2)]


def calculate_euclidean_distance(point_1: List[float], point_2: List[float]) -> float:
    """
    Calculate the Euclidean distance between two points.

    :param point_1: The first point.
    :param point_2: The second point.
    :return: A float value that represents the Euclidean distance between the two points.
    """
    return np.linalg.norm(np.array(point_1) - np.array(point_2))


def calculate_translation_vector(point_1: List[float], point_2: List[float]):
    """
    Calculate the translation vector between two points.

    :param point_1: The first point.
    :param point_2: The second point.
    :return: A list of float values that represent the translation vector between the two points.
    """
    return [p2 - p1 for p1, p2 in zip(point_1, point_2)]


def calculate_angle_between_quaternions_and_check(quat_1: List[float], quat_2: List[float], threshold: float) -> bool:
    """
    Calculate the angle between two quaternions and checks if it is small.

    :param quat_1: The first quaternion.
    :param quat_2: The second quaternion.
    :param threshold: The threshold for the angle between the two quaternions to be considered as small.
    :return: A boolean value that represents the condition for the angle between the two quaternions
     to be considered as small.
    """
    quat_diff_angle = calculate_angle_between_quaternions(quat_1, quat_2)
    return quat_diff_angle <= threshold


def calculate_angle_between_quaternions(quat_1: List[float], quat_2: List[float]) -> float:
    """
    Calculate the angle between two quaternions.

    :param quat_1: The first quaternion.
    :param quat_2: The second quaternion.
    :return: A float value that represents the angle between the two quaternions.
    """
    quat_diff = calculate_quaternion_difference(quat_1, quat_2)
    quat_diff_angle = 2 * np.arctan2(np.linalg.norm(quat_diff[0:3]), quat_diff[3])
    return quat_diff_angle


def calculate_quaternion_difference(quat_1: List[float], quat_2: List[float]) -> List[float]:
    """
    Calculate the quaternion difference.

    :param quat_1: The quaternion of the object at the first time step.
    :param quat_2: The quaternion of the object at the second time step.
    :return: A list of float values that represent the quaternion difference.
    """
    quat_diff = quaternion_multiply(quaternion_inverse(quat_1), quat_2)
    return quat_diff
