import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle, islice
import itertools
from mlxtend.plotting import plot_decision_regions



def plot_loss_function(lam=2):
    
    x = np.arange(0.00, 1, 0.01)
    y = np.arange(0.00, 1, 0.01)
    
    test_error, train_error = np.meshgrid(x, y, sparse=True)
    z = np.array(train_error + lam*(test_error - train_error)**2)
    
    
    contours = plt.contour(x,y,z, 20, colors='black')
    plt.clabel(contours, inline=True, fontsize=8)

    plt.imshow(z, extent=[0, 1, 0, 1], origin='lower',
               cmap='RdGy', alpha=0.5)

    plt.plot(x, x, ':r') # dashdot black
    
    plt.xlabel('Test error')
    
    plt.ylabel('Train error')

    plt.colorbar()
    
def plot_toy_set(X, y, tittle):
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                 '#f781bf', '#a65628', '#984ea3',
                                                 '#999999', '#e41a1c', '#dede00']),
                                          int(max(y) + 1))))
    plt.title(tittle)
    plt.scatter(X[:, 0], X[:, 1],color=colors[y])
    plt.show()
    
def custom_decision_region_plot(X, y, model, title):
    # Specify keyword arguments to be passed to underlying plotting functions
    scatter_kwargs = {'s': 120, 'edgecolor': None, 'alpha': 0.7}
    contourf_kwargs = {'alpha': 0.2}
    scatter_highlight_kwargs = {'s': 60, 'label': 'Test data', 'alpha': 0.7, 'c': 'red'}

    # Plotting decision regions       
    plot_decision_regions(X=X, 
                          y=y,
                          clf=model, 
                          legend=2, 
                          scatter_kwargs=scatter_kwargs,
                          contourf_kwargs=contourf_kwargs,
                          scatter_highlight_kwargs=scatter_highlight_kwargs)
    plt.title(title)
    plt.show()
    
def learners_decision_regions_plot(X, y, learners, zoom=1):
    # Plotting Decision Regions

    num_rows = np.ceil(len(learners)/2).astype(int)

    fig, axs = plt.subplots(num_rows,2,figsize=(max(num_rows*2, 16), max(num_rows*5, 16)),  constrained_layout=True)

    for clf_learner, grd in zip(learners,
                             itertools.product(range(num_rows), [0, 1])):
        try:
            ax = axs[grd[0]][grd[1]]
        except:
            ax = axs[grd[1]]

        # Specify keyword arguments to be passed to underlying plotting functions
        scatter_kwargs = {'s': 120, 'edgecolor': None, 'alpha': 0.7}
        contourf_kwargs = {'alpha': 0.2}
        scatter_highlight_kwargs = {'s': 60, 'label': 'Test data', 'alpha': 0.7, 'c': 'red'}

        # Plotting decision regions       
        plot_decision_regions(X=X[np.ix_(clf_learner['data']['train_indexes'], clf_learner['data']['selected_features'])], 
                              y=y[clf_learner['data']['train_indexes']],
                              clf=clf_learner['learner'],
                              legend=2, 
                              ax = ax, 
                              zoom_factor=zoom,
                              
#                             X=X[np.append(clf_learner['data']['train_indexes'],clf_learner['data']['oob_indexes'])], 
#                             y=y[np.append(clf_learner['data']['train_indexes'],clf_learner['data']['oob_indexes'])],
#                             X_highlight=X[clf_learner['data']['oob_indexes']],

                              scatter_kwargs=scatter_kwargs,
                              contourf_kwargs=contourf_kwargs,
                              scatter_highlight_kwargs=scatter_highlight_kwargs)

        ax.set_title(clf_learner['learner'])
        ax.set_xlim([int(X[:, 0].min() - 2) , int(X[:, 0].max() + 2)])
        ax.set_ylim([int(X[:, 1].min() - 2) , int(X[:, 1].max() + 2)])

    plt.show()
    
    
def custom_decision_region_plot_v2(X, y, clf_learner, title='', zoom=1):
    # Specify keyword arguments to be passed to underlying plotting functions
    scatter_kwargs = {'s': 120, 'edgecolor': None, 'alpha': 0.7}
    contourf_kwargs = {'alpha': 0.2}
    scatter_highlight_kwargs = {'s': 60, 'label': 'Test data', 'alpha': 0.7, 'c': 'red'}

    # Plotting decision regions       
    plot_decision_regions(X=X[np.ix_(clf_learner['data']['train_indexes'], clf_learner['data']['selected_features'])], 
                        y=y[clf_learner['data']['train_indexes']],
                        clf=clf_learner['learner'],     
                        legend=2, 
                        zoom_factor=zoom,
                        scatter_kwargs=scatter_kwargs,
                        contourf_kwargs=contourf_kwargs,
                        scatter_highlight_kwargs=scatter_highlight_kwargs)
    plt.xlim([int(X[:, 0].min() - 2) , int(X[:, 0].max() + 2)])
    plt.ylim([int(X[:, 1].min() - 2) , int(X[:, 1].max() + 2)])
    plt.title(title)
    plt.show()
    
def WinTieLoss(wtl_df, leyend=True,saving_path=None):
    
    sns.set_theme(style="whitegrid")

    sns.color_palette("coolwarm", as_cmap=True)

    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=(15, 6))

    # Plot Losses
    sns.barplot(x="Losses", y=wtl_df.index, data=wtl_df,
                label="Losses", color="indianred")

    # Plot Ties
    sns.barplot(x="Ties", y=wtl_df.index, data=wtl_df,
                label="Ties", color="gold")


    # Plot Wins
    sns.barplot(x="Wins", y=wtl_df.index, data=wtl_df,
                label="Wins", color="seagreen")

    # Critical value 90%
    n_df = wtl_df.max().max()
    nc = int(n_df/2 + 1.645*(np.sqrt(n_df)/2))
    plt.axvline(nc, color='black', linewidth=2.5, linestyle='dashdot', label='alpha 0.10')

    # Critical value 95%
    n_df = wtl_df.max().max()
    nc = int(n_df/2 + 1.96*(np.sqrt(n_df)/2))
    plt.axvline(nc, color='midnightblue', linewidth=2.5, linestyle='dashdot', label='alpha 0.05')

    if leyend==True:
        # Add a legend and informative axis label
        ax.legend(ncol=6, loc="center", frameon=False,  bbox_to_anchor=(.5, 1.05))
    else:
         saving_path = saving_path+'_no_leyend'
         
    ax.set(xlim=(0, n_df), ylabel="",
        xlabel="Number of Datasets")
    sns.despine(left=True, bottom=True)
    
    if saving_path is not None:
        plt.savefig(f'{saving_path}.png')