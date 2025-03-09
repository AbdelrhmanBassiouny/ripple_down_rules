from __future__ import annotations

from typing import List

from sqlalchemy import create_engine, ForeignKey, select
from sqlalchemy.orm import DeclarativeBase, MappedAsDataclass, Mapped, Session, relationship, aliased, declared_attr
from sqlalchemy.testing.schema import mapped_column


class Base(MappedAsDataclass, DeclarativeBase):

    @declared_attr.directive
    def __tablename__(cls) -> str:
        return cls.__name__


class PolymorphicIdentityMixin:

    @declared_attr.directive
    def __mapper_args__(cls):
        return {
            "polymorphic_identity": cls.__tablename__,
        }


class HasPart(Base):
    left_id: Mapped[int] = mapped_column(ForeignKey("PhysicalObject.id"), primary_key=True, init=False, repr=False)
    right_id: Mapped[int] = mapped_column(ForeignKey("PhysicalObject.id"), primary_key=True, init=False, repr=False)
    left: Mapped[PhysicalObject] = relationship(back_populates="parts", foreign_keys=[left_id])
    right: Mapped[PhysicalObject] = relationship(back_populates="part_of", foreign_keys=[right_id])


class PhysicalObject(Base):
    id: Mapped[int] = mapped_column(primary_key=True, init=False)
    name: Mapped[str]
    parts: Mapped[List[HasPart]] = relationship(init=False, back_populates="left", foreign_keys=[HasPart.left_id],
                                                repr=False)
    part_of: Mapped[List[HasPart]] = relationship(init=False, back_populates="right", foreign_keys=[HasPart.right_id],
                                                  repr=False)

    type: Mapped[str] = mapped_column(init=False)

    @declared_attr.directive
    def __mapper_args__(cls):
        return {
            "polymorphic_on": "type",
            "polymorphic_identity": cls.__tablename__,
        }


class IsPerishable(MappedAsDataclass):
    perishable: Mapped[bool] = mapped_column(init=False, default=True)


class IsLiquid(MappedAsDataclass):
    liquid: Mapped[bool] = mapped_column(init=False, default=True)


class Cabinet(PolymorphicIdentityMixin, PhysicalObject):
    id: Mapped[int] = mapped_column(ForeignKey(PhysicalObject.id), primary_key=True, init=False)


class Drawer(PolymorphicIdentityMixin, PhysicalObject):
    id: Mapped[int] = mapped_column(ForeignKey(PhysicalObject.id), primary_key=True, init=False)


class Kitchen(PolymorphicIdentityMixin, PhysicalObject):
    id: Mapped[int] = mapped_column(ForeignKey(PhysicalObject.id), primary_key=True, init=False)


class Food(PolymorphicIdentityMixin, PhysicalObject):
    id: Mapped[int] = mapped_column(ForeignKey(PhysicalObject.id), primary_key=True, init=False)


class Milk(Food, IsPerishable, IsLiquid):
    id: Mapped[int] = mapped_column(ForeignKey(Food.id), primary_key=True, init=False)


def get_all_parts(session, object_id):
    part_of_alias = aliased(HasPart)
    physical_object_alias = aliased(PhysicalObject)

    cte = (
        select(HasPart.right_id.label('id'))
        .where(HasPart.left_id == object_id)
        .cte(name='parts', recursive=True)
    )

    cte = cte.union_all(
        select(part_of_alias.right_id)
        .join(cte, cte.c.id == part_of_alias.left_id)
    )

    query = (
        select(physical_object_alias)
        .join(cte, cte.c.id == physical_object_alias.id)
    )

    return session.scalars(query).all()


def main():
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)

    session = Session(engine)
    kitchen = Kitchen('kitchen')
    cabinet = Cabinet('cabinet')
    drawer = Drawer('drawer')

    milk = Milk(name='milk')

    print(milk)

    hp1 = HasPart(left=kitchen, right=cabinet)

    cabinet.parts.append(HasPart(left=cabinet, right=drawer))
    session.add_all([kitchen, cabinet, drawer, hp1])
    session.commit()

    result = session.scalars(select(PhysicalObject)).all()
    print([isinstance(r, Kitchen) for r in result])
    r = get_all_parts(session, kitchen.id)
    print(r)


if __name__ == '__main__':
    main()
