import pathlib
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap

# Define the helper functions

def rotate_color(c):
  cc = c[-1], *c[:-1]
  return tuple(((a+b)%1.0 for a, b in zip(c, cc)))

def cont_plot( ax, pts, color=(.0, .0, .0)
             , mcolor=(1.0, .0, .0), mstyle='o', msz=1
             , fill=False ):
  #ax.plot( *zip(*pts), linewidth=1, color=color
  #       , marker=mstyle, ms=msz, mec=mcolor, mfc=mcolor )
  ax.fill(*zip(*pts), fill=fill, linewidth=1, color=color)
  ax.scatter(*zip(*pts), marker=mstyle, s=msz, color=mcolor)

def asym_to_sym(asym_dist_mat):
  """
  Convert an asymmetric distance matrix to a symmetric one.
  """
  return np.max(np.stack([asym_dist_mat, asym_dist_mat.T]), axis=0)

def dst_mat_to_coords(dst_mat):
  """
  Convert a distance matrix to 2D coordinates using MDS.
  """
  embedding = MDS(n_components=2, dissimilarity='precomputed', normalized_stress='auto', n_init=4)
  return embedding.fit_transform(dst_mat)

def plot_xy_points( points, save_path=None, title=None
                  , color=(.0, .0, .0), mcolor=(1.0, .0, .0)):
  fig, ax = plt.subplots()

  #ax.scatter(*zip(*points), s=6)
  cont_plot(ax, points, color=color, mcolor=mcolor)

  ax.set_aspect('equal')
  ax.set_xticks([])
  ax.set_yticks([])

  if title: ax.set_title(title)

  if save_path is not None:
    fig.savefig(save_path)
    plt.close(fig)
  else: return fig

def dst_mat_compare_plot( orig_dm, orig_contour
                        , preproc_dm, preproc_contour
                        , recon_dm, recon_contour
                        , save_path=None ):
  fig, ax = plt.subplots(2, 3)

  ax[0, 0].matshow(orig_dm)
  ax[0, 1].matshow(preproc_dm)
  ax[0, 2].matshow(recon_dm)
  #ax[1, 0].scatter(*zip(*orig_contour), s=6, color=(.0, .0, 1.0))
  cont_plot(ax[1, 0], orig_contour)
  #ax[1, 1].scatter(*zip(*preproc_contour), s=6, color=(.5, .0, 1.0))
  cont_plot(ax[1, 1], preproc_contour, (.5, .0, 1.0))
  #ax[1, 2].scatter(*zip(*recon_contour), s=6, color=(1.0, .5, .0))
  cont_plot(ax[1, 2], recon_contour, (1.0, .5, .0))

  ax[0, 0].set_title('original\ndistance matrix')
  ax[0, 1].set_title('preprocessed\ndistance matrix')
  ax[0, 2].set_title('reconstructed\ndistance matrix')
  ax[1, 0].set_title('original\ncontour')
  ax[1, 1].set_title('preprocessed\ncontour')
  ax[1, 2].set_title('reconstructed\ncontour')

  for a in ax.flatten():
    a.set_aspect('equal')
    a.set_xticks([])
    a.set_yticks([])

  if save_path is not None:
    fig.savefig(save_path)
    plt.close(fig)
  else: return fig

def mean_shape_rprt(mean_shape, save_path=None):
  save_path=pathlib.Path(save_path)
  fig, ax = plt.subplots(len(mean_shape), 3)
  figs = []
  for i, (lbl, (_, recon_x)) in enumerate(mean_shape.items()):
    fig_n, ax_n = plt.subplots(1, 3)
    ##
    ax[i, 0].text( 0.5, 0.5, f'{lbl}'
                 , horizontalalignment='center'
                 , verticalalignment='center'
                 , transform=ax[i, 0].transAxes )
    ax_n[0].text( 0.5, 0.5, f'{lbl}'
                , horizontalalignment='center'
                , verticalalignment='center'
                , transform=ax_n[0].transAxes )
    ##
    recon_x = recon_x.squeeze().to('cpu')
    ax[i, 1].imshow(recon_x)
    ax_n[1].imshow(recon_x)
    cont = dst_mat_to_coords(asym_to_sym(recon_x))
    #ax[i, 2].scatter(*zip(*cont), s=1, color=(.0, .0, 1.0))
    cont_plot(ax[i, 2], cont)
    #ax_n[2].scatter(*zip(*cont), s=1, color=(.0, .0, 1.0))
    col = (.0, .0, 1.0)
    cont_plot(ax_n[2], cont)

    for a in ax[i]:
      a.set_aspect('equal')
      a.set_xticks([])
      a.set_yticks([])
    for a in ax_n:
      a.set_aspect('equal')
      a.set_xticks([])
      a.set_yticks([])

    figs.append((fig_n, lbl, ax_n))

  if save_path is not None:
    fig.savefig(save_path)
    plt.close(fig)
    for f, lbl, _ in figs:
      f.savefig(f'{save_path.parent/save_path.stem}_{lbl}{save_path.suffix}')
      plt.close(f)
  else: return fig, figs

def dst_mat_compare_plot_table( entries
                              , save_path=None ):
  if len(entries) == 0: return
  fig, ax = plt.subplots(len(entries), 7)

  for i, (_ , og_m, preproc_m, recon_m
            , og_cont, preproc_cont, recon_cont
            , recon_loss) in enumerate(entries):
    ax[i, 0].imshow(og_m)
    ax[i, 1].imshow(preproc_m)
    ax[i, 2].imshow(recon_m)
    #ax[i, 3].scatter(*zip(*og_cont), s=1, color=(.0, .0, 1.0))
    cont_plot(ax[i, 3], og_cont)
    #ax[i, 4].scatter(*zip(*preproc_cont), s=1, color=(.5, .0, 1.0))
    cont_plot(ax[i, 4], preproc_cont, (.5, .0, 1.0))
    #ax[i, 5].scatter(*zip(*recon_cont), s=1, color=(1.0, .5, .0))
    cont_plot(ax[i, 5], recon_cont, (1.0, .5, .0))
    ax[i, 6].text( 0.5, 0.5, f'{recon_loss:.4f}'
                 , horizontalalignment='center'
                 , verticalalignment='center'
                 , transform=ax[i, 6].transAxes )

    for a in ax.flatten():
      #a.set_aspect('equal')
      a.set_aspect('equal', adjustable='datalim')
      a.set_xticks([])
      a.set_yticks([])

  ax[0, 0].set_title('original\ndistance matrix', fontsize=8, rotation=0)
  ax[0, 1].set_title('preprocessed\ndistance matrix', fontsize=8, rotation=0)
  ax[0, 2].set_title('reconstructed\ndistance matrix', fontsize=8, rotation=0)
  ax[0, 3].set_title('original\ncontour', fontsize=8, rotation=0)
  ax[0, 4].set_title('preprocessed\ncontour', fontsize=8, rotation=0)
  ax[0, 5].set_title('reconstructed\ncontour', fontsize=8, rotation=0)
  ax[0, 6].set_title('reconstruction\nloss', fontsize=8, rotation=0)

  fig.subplots_adjust(wspace=0.025, hspace=0.05)
  fig.tight_layout()

  if save_path is not None:
    fig.savefig(save_path)
    plt.close(fig)
  else: return fig

################################################################################

# dimensionality reduction

def dimensionality_reduction_plot( feats, lbls, alg
                                 , outfile
                                 , title=None
                                 , tsne_rnd_seed=42, tsne_perplexity=5
                                 , umap_n_neighbors=20, umap_min_dist=0.1 ):
  emb = None
  match alg.lower():
    case 'pca':
      emb = PCA(n_components=2).fit_transform(feats)
    case 'tsne':
      reducer = TSNE( n_components=2
                    , random_state=tsne_rnd_seed, perplexity=tsne_perplexity )
      emb = reducer.fit_transform(feats)
    case 'umap':
      reducer = umap.UMAP( n_components=2
                         , n_neighbors=umap_n_neighbors
                         , min_dist=umap_min_dist )
      emb = reducer.fit_transform(feats)

  mks_cycle = itertools.cycle([ 'o', '^', 's', 'X', 'P', 'v', '8', '<', '>'
                              , ',', '.', 'p', '*', 'h', 'H', 'D', 'd' ])
  unique_sorted_lbls = sorted(set(lbls))
  ax = sns.scatterplot( x=emb[:, 0], y=emb[:, 1]
                      , hue=lbls, hue_order=unique_sorted_lbls, style=lbls
                      , markers=[next(mks_cycle) for _ in unique_sorted_lbls]
                      , palette='colorblind' )
  if title: ax.set_title(title, fontsize=24)
  ax.set_xlabel(f'{alg} component 1')
  ax.set_ylabel(f'{alg} component 2')
  ax.get_figure().savefig(outfile)
  ax.get_figure().clf()
