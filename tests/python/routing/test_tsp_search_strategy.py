from examples.python.optimization import tsp_search_strategy as tsp


def test_choco_search_strategy_reduces_the_tsp_search_tree():
    auto = tsp.solve(False)
    guided = tsp.solve(True)

    assert auto.objective == guided.objective == 244
    assert guided.nodes < auto.nodes
