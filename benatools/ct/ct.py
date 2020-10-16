import pydicom
import scipy.ndimage
import numpy as np
import cv2
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import morphology
from skimage import measure
import matplotlib.pyplot as plt
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.figure_factory as ff
from sklearn.cluster import KMeans
import vtk
from vtk.util import numpy_support
import os
from plotly.graph_objs import *

def load_scan(paths, library='vtk', resample_scan=True):
    """
    Load a scan

    Parameters
    ----------
    library : str
        engine to be used to load scan
    resample_scan : bool
        whether to resample or not a scan

    Returns
    -------
    ndarray
        numpy array containing all the CT scan slices

    Raise
    -----
    library not supported
    """
    if library == 'vtk':
        return load_vtk(paths, resample_scan=resample_scan)
    if library == 'pydicom':
        return load_pydicom(paths, resample_scan=resample_scan)
    raise (f'Library {library} not supported')

def load_pydicom(paths, resample_scan=True, return_spacing=False):
    """
    Function to read all DICOM files belonging to a scan. The functions sorts the slices in order.

    Parameters
    ----------
    paths : list of str
        list of paths to read. Normally you should use glob to get the files
    return_thickness : bool
        return slice thickness

    Returns
    -------
    List of slices sorted by Instance Number
    """
    slices, spacing = load_slices(paths, return_spacing=True)
    hu_scan = get_pixels_hu(slices)

    if resample_scan:
        hu_scan = resample(hu_scan, scan_spacing=spacing)

    if return_spacing:
        return hu_scan, spacing

    return hu_scan


def load_slices(paths, return_spacing=False):
    """
    Function to read all DICOM files belonging to a scan. The functions sorts the slices in order.
    Parameters
    ----------
    paths : list of str
        list of paths to read. Normally you should use glob to get the files
    return_thickness : bool
        return slice thickness

    Outputs
    -------
    slices :
        List of slices sorted by Instance Number
    """
    slices = [pydicom.read_file(path) for path in paths]
    slices.sort(key=lambda x: int(x.InstanceNumber), reverse=True)
    try:
        slice_thickness = np.median([np.abs(slices[i].ImagePositionPatient[2] - slices[i + 1].ImagePositionPatient[2]) for i in range(len(slices) - 1)])
    except:
        slice_thickness = np.median([np.abs(slices[0].SliceLocation - slices[1].SliceLocation) for i in range(len(slices) - 1)])

    for s in slices:
        s.SliceThickness = slice_thickness

    if return_spacing:
        return slices, np.array([slice_thickness] + list(slices[0].PixelSpacing))

    return slices


def get_pixels_hu(scans):
    """
    Function that converts a list of scans into a numpy array converted to HU scale
        Inputs:
            scans: List of sorted scans
        Output:
            numpy array of scans of shape H x W x D
    """
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for n in range(len(scans)):

        intercept = scans[n].RescaleIntercept
        slope = scans[n].RescaleSlope

        if slope != 1:
            image[n] = slope * image[n].astype(np.float64)
            image[n] = image[n].astype(np.int16)

        image[n] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def resample(scan_arr, scan=None, scan_spacing=None):
    """
    Resample a scan in array format, to adjust it to its pixel spacing and thickness
        Input:
            scan_arr: CT Scan in numpy array format
            scan =
    """
    # Determine current pixel spacing
    spacing = scan_spacing if scan_spacing else  np.array([scan[0].SliceThickness] + list(scan[0].PixelSpacing), dtype=np.float32)

    resize_factor = spacing
    new_real_shape = scan_arr.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / scan_arr.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(scan_arr, real_resize_factor, mode='nearest')

    return image


def resize_scan(scan, new_shape=(128,128,128)):
    """
    Resize a 3D image using Open CV. The resizing is slice by slice
        Input:
            scan =
            new_shape =
        """
    resized = np.stack([cv2.resize(img, (new_shape[1], new_shape[2])) for img in scan])
    resized = np.stack([cv2.resize(resized[:,:,i], (new_shape[0], new_shape[1])) for i in range(resized.shape[2])], axis=-1 )
    return resized


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

def air_removal_mask(dilation):
    labels = measure.label(dilation)
    label_vals = np.unique(labels)
    if labels[0,0] == labels[-1, -1]:
        upper_cut = (labels==labels[0,0])
        mask = np.abs(upper_cut*1 -1)
    else:
        upper_cut = (labels == labels[0,0])
        lower_cut = (labels == labels[-1,-1])
        mask = np.abs((upper_cut + lower_cut )*1 -1)
    return mask


def make_lungmask(img, display=False, mean=None, std=None):
    row_size = img.shape[0]
    col_size = img.shape[1]

    # re-scale
    mean = np.mean(img) if mean is None else mean
    std = np.std(img) if std is None else std
    img = img - mean
    img = img / std
    # Find the average pixel value near the lungs
    # to renormalize washed out images
    middle = img[int(col_size / 5):int(col_size / 5 * 4), int(row_size / 5):int(row_size / 5 * 4)]
    mean = np.mean(middle)
    max = np.max(img)
    min = np.min(img)
    # To improve threshold finding, I'm moving the
    # underflow and overflow on the pixel spectrum
    img[img == max] = mean
    img[img == min] = mean
    #
    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
    #
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle, [np.prod(middle.shape), 1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img < threshold, 1.0, 0.0)  # threshold the image

    # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.
    # We don't want to accidentally clip the lung.

    eroded = morphology.erosion(thresh_img, np.ones([3, 3]))
    dilation = morphology.dilation(eroded, np.ones([8, 8]))

    labels = measure.label(dilation)  # Different labels are displayed in different colors
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2] - B[0] < row_size / 10 * 9 and B[3] - B[1] < col_size / 10 * 9 and B[0] > row_size / 5 and B[
            2] < col_size / 5 * 4:
            good_labels.append(prop.label)
    mask = np.ndarray([row_size, col_size], dtype=np.int8)
    mask[:] = 0

    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask
    #
    for N in good_labels:
        mask = mask + np.where(labels == N, 1, 0)
    mask = morphology.dilation(mask, np.ones([10, 10]))  # one last dilation [10,10]

    mask = dilation.astype('int16') * air_removal_mask(dilation)

    if (display):
        fig, ax = plt.subplots(3, 2, figsize=[12, 12])
        ax[0, 0].set_title("Original")
        ax[0, 0].imshow(img, cmap='gray')
        ax[0, 0].axis('off')

        ax[0, 1].set_title("Threshold")
        ax[0, 1].imshow(thresh_img, cmap='gray')
        ax[0, 1].axis('off')

        ax[1, 0].set_title("After Erosion and Dilation")
        ax[1, 0].imshow(dilation, cmap='gray')
        ax[1, 0].axis('off')

        ax[1, 1].set_title("Color Labels")
        ax[1, 1].imshow(labels)
        ax[1, 1].axis('off')

        ax[2, 0].set_title("Final Mask")
        ax[2, 0].imshow(mask, cmap='gray')
        ax[2, 0].axis('off')

        ax[2, 1].set_title("Apply Mask on Original")
        ax[2, 1].imshow(mask * img, cmap='gray')
        ax[2, 1].axis('off')

        plt.show()
    return mask * img


########## VTK Library ########################

def load_vtk_dir(path):
    reader = vtk.vtkDICOMImageReader()
    reader.SetDirectoryName(path)
    reader.Update()
    _extent = reader.GetDataExtent()
    ConstPixelDims = [_extent[1] - _extent[0] + 1, _extent[3] - _extent[2] + 1, _extent[5] - _extent[4] + 1]

    # Load spacing values
    ConstPixelSpacing = reader.GetPixelSpacing()

    # Get the 'vtkImageData' object from the reader and get the 'vtkPointData' object from the 'vtkImageData' object
    pointData = reader.GetOutput().GetPointData()
    # Ensure that only one array exists within the 'vtkPointData' object
    assert (pointData.GetNumberOfArrays() == 1)
    # Get the `vtkArray` (or whatever derived type) which is needed for the `numpy_support.vtk_to_numpy` function
    arrayData = pointData.GetArray(0)
    # Convert the `vtkArray` to a NumPy array and reshape the NumPy array to 3D using 'ConstPixelDims' as a 'shape'
    arr = numpy_support.vtk_to_numpy(arrayData).reshape(ConstPixelDims, order='F')

    return arr, reader

def load_vtk_file(path):
    reader = vtk.vtkDICOMImageReader()
    if os.path.isdir(path):
        reader.SetDirectoryName(path)
    else:
        reader.SetFileName(path)
    reader.Update()
    _extent = reader.GetDataExtent()
    ConstPixelDims = [_extent[1] - _extent[0] + 1, _extent[3] - _extent[2] + 1, _extent[5] - _extent[4] + 1]

    arrayData = reader.GetOutput().GetPointData().GetArray(0)
    ArrayDicom = numpy_support.vtk_to_numpy(arrayData)
    ArrayDicom = ArrayDicom.reshape((reader.GetHeight(), reader.GetWidth()), order='F')
    return ArrayDicom, reader


def load_vtk(paths, resample_scan=True, return_spacing=False, sort_paths=False):
    if isinstance(paths, list):
        if sort_paths:
            paths.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]), reverse=True)
        slices = [load_vtk_file(path) for path in paths]
        scan = np.stack([s[0] for s in slices]).astype(np.int16)
        pixel_spacing = np.array([s[1].GetPixelSpacing() for s in slices])
        pixel_spacing = np.median(pixel_spacing, axis=0)
    else:
        if os.path.isdir(paths):
            scan, reader = load_vtk_dir(paths)
            pixel_spacing = reader.GetPixelSpacing()
        else:
            raise("No valid paths format")

    thickness = pixel_spacing[2]
    pixel_spacing = pixel_spacing[0]

    if resample_scan:
        scan = resample(scan, scan_spacing=np.array([thickness, pixel_spacing, pixel_spacing]))

    if return_spacing:
        return scan, np.array([thickness, pixel_spacing, pixel_spacing])

    return scan