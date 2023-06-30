from amplify_bench.problem.base import gen_problem


def test_gen_problem():
    problem = gen_problem("Tsp", "random50", seed=0, constraint_weight=0.5)

    assert type(problem).__name__ == "Tsp"
    problem_info = problem.get_input_parameter()

    assert problem_info.get("class") == "Tsp"
    assert problem_info.get("instance") == "random50"
    problem_info_parameters = problem_info.get("parameters")
    assert problem_info_parameters.get("seed") == 0
    assert problem_info_parameters.get("constraint_weight") == 0.5

    assert type(problem.get_id()) == str
