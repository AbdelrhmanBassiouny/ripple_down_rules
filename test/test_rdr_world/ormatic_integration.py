import sqlalchemy
from sqlalchemy import select, ForeignKeyConstraint, Table, MetaData, inspect
from sqlalchemy.sql.ddl import DropConstraint, DropTable

from test.conf.world.handles_and_containers import HandlesAndContainersWorld
from ripple_down_rules.orm_interface import *
from ormatic.dao import to_dao, get_dao_class


def drop_all_tables_ignoring_constraints(engine):
    """
    Drops all tables in the database connected by the given engine,
    explicitly dropping foreign key constraints first to avoid issues.
    """
    conn = engine.connect()
    trans = conn.begin()
    inspector = inspect(engine)

    # Get all table names
    table_names = inspector.get_table_names()

    # If using schemas, you might need to specify them:
    # schemas = inspector.get_schema_names()
    # for schema in schemas:
    #     table_names.extend(inspector.get_table_names(schema=schema))

    meta = MetaData()
    tables = []
    all_fkeys = []

    # Reflect each table and collect foreign keys
    for table_name in table_names:
        # Reflect table with just name and meta
        current_table = Table(table_name, meta)
        inspector.reflect_table(current_table, None)  # reflect columns, PKs, and FKs

        fkeys = []
        for fkey in inspector.get_foreign_keys(table_name):
            # Create a ForeignKeyConstraint object to represent the constraint
            # This is important for dropping it later
            if fkey.get("name"):  # Ensure the foreign key has a name
                fkeys.append(ForeignKeyConstraint((), (), name=fkey["name"]))

        # Recreate the table object with its foreign keys for dropping
        # This step is crucial for `DropConstraint` to work
        tables.append(Table(table_name, meta, *fkeys, extend_existing=True))
        all_fkeys.extend(fkeys)

    # Drop all foreign key constraints first
    for fkey in all_fkeys:
        try:
            conn.execute(DropConstraint(fkey))
            print(f"Dropped foreign key constraint: {fkey.name}")
        except Exception as e:
            print(f"Could not drop foreign key constraint {fkey.name}: {e}")

    # Then drop all tables
    for table in tables:
        try:
            conn.execute(DropTable(table))
            print(f"Dropped table: {table.name}")
        except Exception as e:
            print(f"Could not drop table {table.name}: {e}")

    trans.commit()
    conn.close()
    print("All tables and associated constraints dropped successfully (or attempted).")


def main():
    connection_string = "rdr@localhost:3306/RDR" # os.getenv("RDR_DATABASE_URL")
    engine = sqlalchemy.create_engine("mysql+pymysql://" + connection_string)
    session = sqlalchemy.orm.Session(engine)
    Base.metadata.create_all(engine)

    # this is the case structure
    case = HandlesAndContainersWorld().create()
    dao = to_dao(case)
    session.add(dao)
    session.commit()

    # do this in the notebook
    result = session.scalar(select(get_dao_class(type(case))))
    # the is the reconstructed case
    reconstructed_case = result.from_dao()
    session.close()
    drop_all_tables_ignoring_constraints(engine)



if __name__ == '__main__':
    main()