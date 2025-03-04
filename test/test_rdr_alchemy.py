import sqlalchemy.orm
from sqlalchemy import select

from ripple_down_rules.datasets import Base, Animal, Species, get_dataset
from ripple_down_rules.datastructures import RDRMode, PromptFor
from ripple_down_rules.experts import Human
from ripple_down_rules.rdr import SingleClassRDR, MultiClassRDR
from ripple_down_rules.prompt import prompt_user_for_expression
from ripple_down_rules.utils import render_tree
from test_rdr import TestRDR


class TestAlchemyRDR:
    session: sqlalchemy.orm.Session

    @classmethod
    def setUpClass(cls):
        zoo = get_dataset(111, TestRDR.cache_file)

        # data (as pandas dataframes)
        X = zoo['features']
        y = zoo['targets']
        # get ids as list of strings

        category_names = ["mammal", "bird", "reptile", "fish", "amphibian", "insect", "molusc"]
        category_id_to_name = {i + 1: name for i, name in enumerate(category_names)}
        X.loc[:, "species"] = [Species(category_id_to_name[i]) for i in y.values.flatten()]

        engine = sqlalchemy.create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session = sqlalchemy.orm.Session(engine)
        session.bulk_insert_mappings(Animal, X.to_dict(orient="records"))
        session.commit()
        cls.session = session

    def test_setup(self):
        r = self.session.scalars(select(Animal)).all()
        assert len(r) == 101
        # user_input = "hair == 1"
        conditions = prompt_user_for_expression(r[0], PromptFor.Conditions, "species", bool, session=self.session)
        print(conditions)
        print(type(conditions))
        print(conditions(r[0]))
        print(type(conditions(r[0])))

    def test_classify_scrdr(self):
        expert = Human(use_loaded_answers=False)

        query = select(Animal)
        result = self.session.scalars(query).all()
        scrdr = SingleClassRDR()
        scrdr.table = Animal
        scrdr.target_column = Animal.species

        cat = scrdr.fit_case(result[0], target=result[0].species, expert=expert, session=self.session)
        assert cat == result[0].species

    def test_fit_scrdr(self):
        use_loaded_answers = True
        draw_tree = True
        filename = "./test_expert_answers" + "/scrdr_expert_answers_fit"
        expert = Human(use_loaded_answers=use_loaded_answers, session=self.session)
        if use_loaded_answers:
            expert.load_answers(filename)

        query = select(Animal)
        result = self.session.scalars(query).all()
        scrdr = SingleClassRDR(session=self.session)
        scrdr.table = Animal
        scrdr.target_column = Animal.species
        targets = [r.species for r in result]
        scrdr.fit(result, targets, expert=expert,
                  animate_tree=draw_tree, session=self.session)

        cat = scrdr.classify(result[50])
        assert cat.pop() == targets[50]

    def test_fit_mcrdr_stop_only(self):
        use_loaded_answers = True
        draw_tree = False
        filename = "./test_expert_answers" + "/scrdr_expert_answers_fit"
        expert = Human(use_loaded_answers=use_loaded_answers, session=self.session)
        if use_loaded_answers:
            expert.load_answers(filename)

        query = select(Animal)
        result = self.session.scalars(query).all()
        mcrdr = MultiClassRDR(session=self.session)
        mcrdr.table = Animal
        mcrdr.target_column = Animal.species
        targets = [r.species for r in result]
        mcrdr.fit(result, targets, expert=expert, animate_tree=draw_tree)

        cats = mcrdr.classify(result[50])
        assert cats[0] == result[50]
        assert len(cats) == 1


tests = TestAlchemyRDR()
tests.setUpClass()
# tests.test_setup()
# tests.test_alchemy_rules()
tests.test_classify_scrdr()
# tests.test_fit_scrdr()
