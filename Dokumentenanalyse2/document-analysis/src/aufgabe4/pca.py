import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d  # IGNORE:unused-import
from matplotlib.patches import FancyArrowPatch
import numpy as np

class PCAExample(object):
    '''
    Einfaches 3D Beispiel fuer die Hauptkomponentenanalyse.
    '''
    
    def __init__(self, samples, target_dim):
        """Initialisiert das PCA Beispiel und berechnet den Unterraum.
        
        Params:
            samples: ndarray mit Trainingsdaten (zeilenweise).
            target_dim: Dimensionalitaet des Unterraums.
        """
        if target_dim < 1 or target_dim > samples.shape[1]:
            raise ValueError('Invalid target dimension')
        
        self.__sub_origin, self.__sub_var, self.__sub_vs = self.__estimate_subspace(samples)
        
        # Implementieren Sie die Dimensionsreduktion
        if target_dim != samples.shape[1]:
            
            self.__sub_var = self.__sub_var[:target_dim]
            self.__sub_vs = self.__sub_vs [:,:target_dim]
            
    def __estimate_subspace(self, samples):
        """Statistische Berechnung des Unterraums.
        
        Params:
            samples: ndarray mit Trainingsdaten (zeilenweise).
        
        Returns: (Ergaenzen Sie die Dokumentation)
            sub_origin:
            sub_var:
            sub_vs:
        """
        n_samples = float(samples.shape[0])
        # Mittelwert der Stichprobe
        samples_mean = np.sum(samples, axis=0) / n_samples
        sub_origin = samples_mean
        # Streuungsmatrix der Stichprobe
        X = samples - samples_mean
        samples_cov = np.dot(X.T, X) / n_samples
        # Eigenwertanalyse der Streuungsmatrix
        sub_var, sub_vs = np.linalg.eig(samples_cov)
        # Sortieren der Eigenvektoren (/ werte) nach Groesse des zugehoehrigen Eigenwerts
        sub_var_ind = np.argsort(sub_var)[::-1]
        sub_var = sub_var[sub_var_ind]
        sub_vs = sub_vs[:, sub_var_ind]
        return sub_origin, sub_var, sub_vs 
    
    def transform_samples(self, samples):
        """Transformiert Daten in den Unterraum
        
        Params:
            samples: ndarray mit zu transformierenden Vektoren (zeilenweise)
            
        Returns:
            ndarray mit transformierten Vektoren (zeilenweise)
        """
        if samples.shape[1] != self.__sub_vs.shape[0]:
            raise ValueError('Samples dimension does not match vector space transformation matrix')
        
        # Ueberlegen Sie, wie man die gesamte samples Matrix in einem transformiert (ohne Schleife)
        
        samples_2d = np.dot(self.__sub_vs.T, samples.T).T
        
        return samples_2d


    def plot_subspace(self, limits, color, linewidth, alpha, ellipsoid=True, coord_system=True):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if ellipsoid:
            self.plot_ellipsoid(center=self.__sub_origin, radii=self.__sub_var,
                            rotation=self.__sub_vs.T, color=color, linewidth=linewidth,
                            alpha=alpha, ax=ax)
        if coord_system:
            self.plot_coordinate_system(center=self.__sub_origin, axes=self.__sub_vs, 
                                        axes_length=self.__sub_var, ax=ax)
        self.set_axis_limits(ax, limits)
        


    @staticmethod
    def set_axis_limits(ax, limits):
        ax.set_xlim(limits[0][0], limits[0][1])
        ax.set_ylim(limits[1][0], limits[1][1])
        if len(limits) == 3:
            ax.set_zlim(limits[2][0], limits[2][1])
            
        
    @staticmethod
    def plot_sample_data(samples, color='b', annotations=None, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        if samples.shape[1] == 2:
            xx = samples[:, 0]
            yy = samples[:, 1]
            ax.scatter(xx, yy, color=color, marker='o', alpha=1.0)
        elif samples.shape[1] == 3:
            xx = samples[:, 0]
            yy = samples[:, 1]
            zz = samples[:, 2]
            ax.scatter(xx, yy, zz, color=color, marker='o', alpha=1.0)
        
        if annotations is not None:
            for sample, annotation in zip(samples, annotations):
                sample_tup = tuple(sample)
                # * operator expands tuple in argument list
                ax.text(*sample_tup,s=annotation)
  
    @staticmethod
    def samples_coordinate_annotations(samples):
        samples_strcoords = [[str(coord) for coord in sample] for sample in samples]
        samples_labels = [' [ %s ]' % ', '.join(sample) for sample in samples_strcoords]
        return samples_labels
  
    @staticmethod
    def plot_ellipsoid(center, radii, rotation, color, linewidth, alpha, ax=None):
        if len(radii) != rotation.shape[0]:
            raise ValueError('Number of radii does not match rotation matrix')
        if len(radii) == 2:
            radii = list(radii) + [0.0]
            rotation = np.vstack((rotation, [0.0] * 3))
        u = np.linspace(0.0, 2.0 * np.pi, 100)
        v = np.linspace(0.0, np.pi, 100)
        x = radii[0] * np.outer(np.cos(u), np.sin(v))
        y = radii[1] * np.outer(np.sin(u), np.sin(v))
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
        for i in range(len(x)):
            for j in range(len(x)):
                [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + center

        # plot
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z, rstride=4, cstride=4, color=color, alpha=alpha, linewidth=linewidth)

    @staticmethod
    def plot_coordinate_system(center, axes, axes_length, ax=None):
        # plot
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        
        xs_center = [center[0]] 
        ys_center = [center[1]]
        zs_center = [center[2]]
        for a, l in zip(axes.T, axes_length):
            p_axis = center + l * a
            xs = list(xs_center)
            ys = list(ys_center)
            zs = list(zs_center)
            xs.append(p_axis[0])
            ys.append(p_axis[1])
            zs.append(p_axis[2])
            arrow = Arrow3D(xs, ys, zs, lw=2)
            ax.add_artist(arrow)

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, _ = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)   
