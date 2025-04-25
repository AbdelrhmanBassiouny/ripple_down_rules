from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ripple_down_rules.datastructures import CaseQuery
from ripple_down_rules.rdr import SingleClassRDR


@dataclass
class WorldEntity:
    world: Optional[World] = field(kw_only=True, default=None, repr=False)


@dataclass
class Body(WorldEntity):
    ...


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


def main():
    world = World()

    handle = Handle(world=world)
    handle_2 = Handle(world=world)
    container_1 = Container(world=world)
    container_2 = Container(world=world)
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
    case_queries = [CaseQuery(view, "correct", mutually_exclusive=True) for view in all_views]
    rdr = SingleClassRDR()
    rdr.fit(case_queries)


if __name__ == '__main__':
    main()
