from counterfactual_xai.add import Add


def test_add():
    assert Add().add_numbers(1, 2) == 3
