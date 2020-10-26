from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from plotly.offline import iplot
from skimage import measure
import matplotlib.pyplot as plt

def plot_3d(image, threshold=700, color="navy"):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    #p = image.transpose(2, 1, 0)

    verts, faces, _, _ = measure.marching_cubes_lewiner(image, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.5)
    mesh.set_facecolor(color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, image.shape[0])
    ax.set_ylim(0, image.shape[1])
    ax.set_zlim(0, image.shape[2])

    plt.show()


def plotly_3d(image, threshold=700, ):
    p = image.transpose(2, 1, 0)
    verts, faces, _, _ = measure.marching_cubes_lewiner(p, threshold)

    x, y, z = zip(*verts)

    print("Drawing")

    # Make the colormap single color since the axes are positional not intensity.
    #    colormap=['rgb(255,105,180)','rgb(255,255,51)','rgb(0,191,255)']
    colormap = ['rgb(236, 236, 212)', 'rgb(236, 236, 212)']

    fig = ff.create_trisurf(x=x, y=y, z=z, plot_edges=False,
                            colormap=colormap,
                            simplices=faces,
                            backgroundcolor='rgb(64, 64, 64)',
                            title="Interactive Visualization")
    iplot(fig)
