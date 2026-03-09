from autograd.nodes import Node
import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt


class Visualizer:
    def __init__(self, base_node: Node) -> None:
        self._base = base_node
        self._build_graph()

    def _build_graph(self) -> None:
        self.graph = nx.DiGraph()

        topo = self._base._build_topologycal_sort()

        for i, node in enumerate(topo):
            name = node.label if node.label else f"Nodo {i}"
            self.graph.add_node(
                node,
                name=name,
                grad=node.grad,
                value=node.value,
                node_type=node._node_type,
            )

        op_id = 0
        for node in topo:
            if node._op:
                op_id_str = f"Op {op_id}: {node._op}"
                self.graph.add_node(op_id_str, node_type="Operation")

                for parent in node.parents:
                    self.graph.add_edge(parent, op_id_str)

                self.graph.add_edge(op_id_str, node)
                op_id += 1
            else:
                if node.parents:
                    for parent in node.parents:
                        self.graph.add_edge(parent, node)

    def draw(self, width=900, height=600):
        """
        Dibuja el grafo computacional de forma interactiva usando Plotly.
        """

        for node in nx.topological_sort(self.graph):
            predecesores = list(self.graph.predecessors(node))
            if not predecesores:
                self.graph.nodes[node]["layer"] = 0
            else:
                self.graph.nodes[node]["layer"] = (
                    max(self.graph.nodes[p].get("layer", 0) for p in predecesores) + 1
                )

        pos = nx.multipartite_layout(self.graph, subset_key="layer", align="vertical")

        edge_x = []
        edge_y = []
        edge_annotations = []

        curvature = 0.15

        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]

            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            dx, dy = x1 - x0, y1 - y0
            px, py = -dy, dx

            c_factor = curvature if x0 <= x1 else -curvature
            cx, cy = mx + px * c_factor, my + py * c_factor

            num_points = 25
            for i in range(num_points + 1):
                t = i / num_points
                bx = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t**2 * x1
                by = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t**2 * y1
                edge_x.append(bx)
                edge_y.append(by)

            edge_x.append(None)
            edge_y.append(None)

            t_tail = 0.85
            ax_data = (
                (1 - t_tail) ** 2 * x0 + 2 * (1 - t_tail) * t_tail * cx + t_tail**2 * x1
            )
            ay_data = (
                (1 - t_tail) ** 2 * y0 + 2 * (1 - t_tail) * t_tail * cy + t_tail**2 * y1
            )

            edge_annotations.append(
                dict(
                    x=x1,
                    y=y1,
                    xref="x",
                    yref="y",
                    ax=ax_data,
                    ay=ay_data,
                    axref="x",
                    ayref="y",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.5,
                    arrowwidth=2,
                    arrowcolor="#7f8c8d",
                    standoff=18,
                )
            )

        data_x, data_y, data_text, data_hover = [], [], [], []
        op_x, op_y, op_text = [], [], []

        for node, d in self.graph.nodes(data=True):
            x, y = pos[node]
            if d.get("node_type") in ("Variable", "Scalar"):
                data_x.append(x)
                data_y.append(y)

                val = d.get("value", 0)
                grad = d.get("grad", 0)
                name = d.get("name", str(node))

                data_text.append(name)
                data_hover.append(
                    f"<b>{name}</b><br>Valor: {val:.4f}<br>Gradiente: {grad:.4f}"
                )

            elif d.get("node_type") == "Operation":
                op_x.append(x)
                op_y.append(y)

                simbolo = str(node).split(": ")[-1] if ": " in str(node) else str(node)
                op_text.append(simbolo)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(color="#7f8c8d", width=2),
                hoverinfo="none",
                name="Conexiones",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=data_x,
                y=data_y,
                mode="markers+text",
                text=data_text,
                textposition="top center",
                hoverinfo="text",
                hovertext=data_hover,
                marker=dict(
                    size=35, color="#aed6f1", line=dict(width=2, color="#2e86c1")
                ),
                name="Datos",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=op_x,
                y=op_y,
                mode="markers+text",
                text=op_text,
                textposition="middle center",
                hoverinfo="text",
                hovertext=op_text,
                marker=dict(
                    symbol="square",
                    size=25,
                    color="#f5b041",
                    line=dict(width=2, color="#d68910"),
                ),
                name="Operaciones",
            )
        )

        fig.update_layout(
            title="Autograd: Computational Graph",
            title_x=0.5,
            width=width,
            height=height,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=20, r=20, t=60),
            annotations=edge_annotations,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
        )

        fig.show()
