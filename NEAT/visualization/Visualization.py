from __future__ import print_function
import warnings
import graphviz


def draw_net(genome, view=False, filename=None, node_names=None, show_disabled=False, node_colors=None, fmt='png'):
    """ This is modified code originally from: https://github.com/CodeReclaimers/neat-python """
    """ Receives a genotype and draws a neural network with arbitrary topology. """
    # Attributes for network nodes.
    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return

    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

    for connection in genome.connections:
        if connection.active or show_disabled:
            input = connection.inputNodeId
            output = connection.outputNodeId

            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))

            style = 'solid' if connection.active else 'dotted'
            color = 'green' if float(connection.weight) > 0 else 'red'
            width = str(0.1 + abs(float(connection.weight / 5.0)))
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.render(filename, view=view)

    return dot