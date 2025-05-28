from datasets import Species, Habitat, load_zoo_dataset
from ripple_down_rules.datastructures.dataclasses import CaseQuery


def pytest_generate_tests(metafunc):

    if metafunc.definition.originalname == "test_should_i_ask_the_expert_for_a_target":
        all_cases, all_targets = load_zoo_dataset("./test_results/zoo")

        possible_case_queries = [CaseQuery(all_cases[0], 'species', (Species,), True),
                                 CaseQuery(all_cases[1], 'habitat', (Habitat,), False)]
        metafunc.parametrize("case_query", possible_case_queries)

        possible_conclusions = [[], None, True, False, set(), {}, {'species': Species.mammal}, Species.mammal,
                                Habitat.land, {'habitat': Habitat.water}, [Habitat.water, Habitat.land],
                                {'species': Species.fish, 'habitat': Habitat.water}]
        metafunc.parametrize("conclusions", possible_conclusions)

        possible_ask_always = [True, False]
        metafunc.parametrize("ask_always", possible_ask_always)

        possible_update_existing = [True, False]
        metafunc.parametrize("update_existing", possible_update_existing)