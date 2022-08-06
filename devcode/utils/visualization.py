import numpy as np
import plotly.graph_objects as go
import plotly.offline as py


def plot_data(X, is_notebook=True):
    fig = go.Figure(data=go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers'))
    fig.update_layout({
        'title': 'MNIST data set after PCA (2 components)'
    })
    # temp = 2.1
    # fig.update_xaxes(range=[-temp, temp])
    # fig.update_yaxes(range=[-temp, temp])

    if is_notebook:
        fig.show("notebook")
    else:
        fig.show()


def plot_voronoi_cells(X, kmeans):
    from scipy.spatial import Voronoi, voronoi_plot_2d
    vor = Voronoi(kmeans.cluster_centers_)

    import matplotlib.pyplot as plt
    voronoi_plot_2d(vor)

    plt.show()  # Voroni cells

    fig = go.Figure()

    colors = [
        '#1f77b4',  # muted blue
        '#ff7f0e',  # safety orange
        '#2ca02c',  # cooked asparagus green
        '#d62728',  # brick red
        '#9467bd',  # muted purple
        '#ffc226',  # chestnut brown
        '#e377c2',  # raspberry yogurt pink
        '#7f7f7f',  # middle gray
        '#bcbd22',  # curry yellow-green
        '#17becf'  # blue-teal
    ]

    for i in range(kmeans.n_clusters):  # add points for each cluster
        fig.add_trace(go.Scatter(
            x=X[kmeans.labels_ == i, 0],
            y=X[kmeans.labels_ == i, 1],
            mode='markers',
            name="partition #{}".format(i + 1),
            opacity=0.7,
            marker=dict(
                size=7,
                color=colors[i],
                #             line = dict(width=1.25,color='#000000')
            )
        ))

    # adding k-means prototypes
    fig.add_trace(go.Scatter(
        x=kmeans.cluster_centers_[:, 0],
        y=kmeans.cluster_centers_[:, 1],
        mode='markers',
        name="K-means prototypes",
        #     marker_symbol='open'
        marker=dict(
            size=15,
            #         color='#2ecc71',
            color='rgba(80, 80, 80, 1)',
            #             color=colors,
            #         color='#0984e3',
            line=dict(
                width=0,
                color='#000')
        )
    ))

    end_x = 100
    line_width = 3
    for i in range(len(vor.ridge_vertices)):
        pair = vor.ridge_vertices[i]

        # proporção entre triângulos
        teste = (
                        vor.vertices[pair[1], 1] - (kmeans.cluster_centers_[vor.ridge_points[i][0], 1] +
                                                    kmeans.cluster_centers_[vor.ridge_points[i][1], 1]) / 2
                ) / (
                        vor.vertices[pair[1], 0] - (kmeans.cluster_centers_[vor.ridge_points[i][0], 0] +
                                                    kmeans.cluster_centers_[vor.ridge_points[i][1], 0]) / 2
                ) * end_x

        if pair[0] >= 0 and pair[1] >= 0:  # voronoi cell limitada
            temp = vor.vertices[pair, :]
            fig.add_trace(go.Scatter(
                x=temp[:, 0],
                y=temp[:, 1],
                mode='lines',
                line=dict(
                    color='rgba(0, 0, 0, 1)',
                    width=line_width,
                    dash='dash'

                ),
                showlegend=False
            ))
        else:  # voronoi cell sem fronteira
            fig.add_trace(go.Scatter(
                x=[
                    vor.vertices[pair[1], 0],
                    - end_x * np.sign(vor.vertices[pair[1], 0] - (kmeans.cluster_centers_[vor.ridge_points[i][0], 0] +
                                                                  kmeans.cluster_centers_[
                                                                      vor.ridge_points[i][1], 0]) / 2)

                ],
                y=[
                    vor.vertices[pair[1], 1],
                    - teste * np.sign(vor.vertices[pair[1], 0] - (kmeans.cluster_centers_[vor.ridge_points[i][0], 0] +
                                                                  kmeans.cluster_centers_[
                                                                      vor.ridge_points[i][1], 0]) / 2)
                ],
                mode='lines',
                line=dict(
                    color='rgba(0, 0, 0, 1)',
                    width=line_width,
                    dash='dash'
                ),
                showlegend=False
            )
            )

    # adjusting limits of plot
    x_max, y_max = np.amax(X, axis=0) * 1.1
    x_min, y_min = np.amin(X, axis=0) * 1.1
    fig.update_xaxes(range=[x_min, x_max])
    fig.update_yaxes(range=[y_min, y_max])

    # fig.update_layout({'showlegend':False});
    fig.update_layout(
        margin=dict(l=20, r=2, t=2, b=0),
        #         paper_bgcolor="LightSteelBlue",
        #         plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        #     xaxis=dict(showgrid=False),
        #     yaxis=dict(showgrid=True),
        autosize=False,
        width=1600 / 2,
        height=900 / 2,
        legend=dict(orientation="h")
    )

    # fig.update_layout(legend=dict(x=.8, y=1))

    fig.show()

    fig.write_image("chap2_kmeans_data_part.pdf", width=900, height=600)

    # creating meshgrid
    n = 200
    x = np.linspace(x_min, x_max, n)
    y = np.linspace(y_min, y_max, n)
    xv, yv = np.meshgrid(x, y)
    x = xv.flatten()
    y = yv.flatten()

    predictions = kmeans.predict(np.column_stack((x, y)))

    fig = go.Figure()

    # paiting feature space partitions
    for i in range(kmeans.n_clusters):  # add points for each cluster
        fig.add_trace(go.Scatter(
            x=x[predictions == i],
            y=y[predictions == i],
            mode='markers',
            name="partition #{}".format(i + 1),
            opacity=0.6,
            #     marker = dict(
            #         size=4.5,
            #         color= kmeans.labels_
            #     )
            marker=dict(
                size=4.5,
                color=colors[i],
            )
        ))

    fig.add_trace(go.Scatter(
        x=kmeans.cluster_centers_[:, 0],
        y=kmeans.cluster_centers_[:, 1],
        mode='markers',
        name='K-means prototypes',
        marker=dict(size=20,
                    color='rgba(80, 80, 80, 1)',
                    )
    ))

    end_x = 100
    line_width = 3
    for i in range(len(vor.ridge_vertices)):
        pair = vor.ridge_vertices[i]

        if pair[0] >= 0 and pair[1] >= 0:  # voronoi cell limitada
            temp = vor.vertices[pair, :]
            fig.add_trace(go.Scatter(
                x=temp[:, 0],
                y=temp[:, 1],
                mode='lines',
                line=dict(
                    color='#000000',
                    width=line_width,
                ),
                showlegend=False
            ))
        else:  # voronoi cell sem fronteira
            # proporção entre triângulos
            teste = (
                            vor.vertices[pair[1], 1] - (kmeans.cluster_centers_[vor.ridge_points[i][0], 1] +
                                                        kmeans.cluster_centers_[vor.ridge_points[i][1], 1]) / 2
                    ) / (
                            vor.vertices[pair[1], 0] - (kmeans.cluster_centers_[vor.ridge_points[i][0], 0] +
                                                        kmeans.cluster_centers_[vor.ridge_points[i][1], 0]) / 2
                    ) * end_x

            fig.add_trace(go.Scatter(
                x=[
                    vor.vertices[pair[1], 0],
                    - end_x * np.sign(vor.vertices[pair[1], 0] - (kmeans.cluster_centers_[vor.ridge_points[i][0], 0] +
                                                                  kmeans.cluster_centers_[
                                                                      vor.ridge_points[i][1], 0]) / 2)

                ],
                y=[
                    vor.vertices[pair[1], 1],
                    - teste * np.sign(vor.vertices[pair[1], 0] - (kmeans.cluster_centers_[vor.ridge_points[i][0], 0] +
                                                                  kmeans.cluster_centers_[
                                                                      vor.ridge_points[i][1], 0]) / 2)
                ],
                mode='lines',
                line=dict(
                    color='#000000',
                    width=line_width,
                ),
                showlegend=False
            )
            )

    # adjusting limits of plot
    fig.update_xaxes(range=[x_min, x_max])
    fig.update_yaxes(range=[y_min, y_max])

    fig.update_layout(
        margin=dict(l=20, r=2, t=2, b=0),
        #         paper_bgcolor="LightSteelBlue",
        #         plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        #     xaxis=dict(showgrid=False),
        #     yaxis=dict(showgrid=True),
        autosize=False,
        width=1600 / 2,
        height=900 / 2,
        legend=dict(orientation="h")
    )

    # fig.update_layout(legend=dict(x=.8, y=1))

    fig.show(renderer="png")

    fig.write_image("chap2_kmeans_data_part_2.pdf", width=700, height=500)


def render_boxplot(results, dataset_name, ks):
    data = [{}] * (len(ks) + 1)

    y_gls = results['GOLS'][dataset_name].values

    data[0] = go.Box(y=y_gls, name=ks[0][2:], marker=dict(color='#2980b9'))

    for i in range(2, len(ks)):
        trace = go.Box(
            y=results['LOLS'][dataset_name][ks[i]].values, name=ks[i][2:], marker=dict(color='#2980b9')
        )
        data[i] = trace

    layout = go.Layout(
        title="Accuracy vs number of clusters [{}]".format(dataset_name),
        showlegend=False,
        yaxis=dict(title="Accuracy on the test set"),
        xaxis=dict(title="Number of clusters")
    )

    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig)


def render_boxplot_train_test(tr_data, ts_data, title, metric_name="Accuracy"):
    train_box = go.Box(y=tr_data, name="Training", boxmean='sd')
    test_box  = go.Box(y=ts_data, name="Testing", boxmean='sd')

    layout = go.Layout(title=title, showlegend=True,
                       yaxis=dict(title=metric_name), xaxis=dict(title="Dataset"))

    fig = go.Figure(data=[train_box, test_box], layout=layout)
    fig.show()


def render_boxplot_with_histogram_train_test(tr_data, ts_data, title, metric_name="Accuracy", bin_size=10):
    import plotly.figure_factory as ff

    # Create distplot with custom bin_size
    fig = ff.create_distplot([tr_data, ts_data], ['Training', 'Testing'], bin_size=bin_size)
    fig.update_layout(title=title, xaxis_title=metric_name, yaxis_title='Frequency')
    fig.show()


def plot_validation_indices(dataset_name, validation_indices):
    data = []
    for index_name, results_vec in validation_indices.items():
        # for validation_index in validation_indices:
        # print(index_name)
        # print(results_vec[dataset_name])
        data.append(go.Scatter(
            x=[i for i in range(2, len(results_vec[dataset_name]) + 2)],
            y=results_vec[dataset_name],
            mode='lines+markers',
            name="{} index".format(index_name)))

    layout = go.Layout(
        title="Indices vs k [{} dataset]".format(dataset_name),
        legend=dict(orientation="h", y=-.05),
        xaxis=dict(title="Number of clusters (k)"),
        yaxis=dict(title="Indices values")
    )

    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig)


def plot_kmeans(kmeans, X):
    data = []
    if X is not None:
        datapoints = go.Scatter(
            x=X[:, 0],
            y=X[:, 1],
            mode='markers',
            name='data',
            marker=dict(
                size=5,
                color='#03A9F4'
            )
        )
        data.append(datapoints)

    kmeans_clusters = go.Scatter(
        x=kmeans.cluster_centers_[:, 0],
        y=kmeans.cluster_centers_[:, 1],
        mode='markers',
        name='kmeans clusters',
        marker=dict(size=10, color='#673AB7')
    )
    data.append(kmeans_clusters)

    layout = go.Layout(
        title="Data + KMeans clusters",
        xaxis=dict(title="$x_1$"),
        yaxis=dict(title="$x_2$"),
    )

    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig)


def plot_datapoints(X, title):
    fig = go.Figure(data=go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers'))
    fig.update_layout({
        'title': title
    })

    fig.show("notebook")



