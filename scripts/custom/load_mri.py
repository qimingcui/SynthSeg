import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt


def load_mri_file(file_path):
    """
    Load an MRI file (.img) using nibabel.
    
    Args:
        file_path (str): Path to the .img file
        
    Returns:
        tuple: (nibabel image object, numpy array of image data)
    """
    try:
        # Load the image
        img = nib.load(file_path)
        
        # Get the image data as a numpy array
        data = img.get_fdata()
        
        return img, data
    
    except Exception as e:
        print(f"Error loading file {file_path}: {str(e)}")
        return None, None


def print_image_info(img, data):
    """
    Print information about the loaded MRI image.
    
    Args:
        img: nibabel image object
        data: numpy array of image data
    """
    if img is None or data is None:
        return
    
    print("\nImage Information:")
    print("-" * 50)
    print(f"Shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Voxel size: {img.header.get_zooms()}")
    print(f"Data range: [{data.min():.2f}, {data.max():.2f}]")
    print(f"Mean value: {data.mean():.2f}")
    print(f"Standard deviation: {data.std():.2f}")


def display_3d_views(data, title="MRI Views"):
    """
    Display axial, sagittal, and coronal views of the MRI data.
    
    Args:
        data: numpy array of image data
        title: title for the figure
    """
    if data is None:
        return
    
    # Get the middle slice for each view
    mid_axial = data.shape[2] // 2
    mid_sagittal = data.shape[0] // 2
    mid_coronal = data.shape[1] // 2
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title)
    
    # Plot axial view
    im1 = ax1.imshow(data[:, :, mid_axial].T, cmap='gray', origin='lower')
    ax1.set_title('Axial View')
    plt.colorbar(im1, ax=ax1)
    
    # Plot sagittal view
    im2 = ax2.imshow(data[mid_sagittal, :, :].T, cmap='gray', origin='lower')
    ax2.set_title('Sagittal View')
    plt.colorbar(im2, ax=ax2)
    
    # Plot coronal view
    im3 = ax3.imshow(data[:, mid_coronal, :].T, cmap='gray', origin='lower')
    ax3.set_title('Coronal View')
    plt.colorbar(im3, ax=ax3)
    
    plt.tight_layout()
    return fig  # Return the figure object instead of showing it directly


def process_mri(file_path, show_info=True, show_views=True):
    """
    Process an MRI file: load it, optionally show information and views.
    
    Args:
        file_path (str): Path to the .img file
        show_info (bool): Whether to print image information
        show_views (bool): Whether to display 3D views
        
    Returns:
        tuple: (nibabel image object, numpy array of image data, matplotlib figure if show_views=True)
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None, None, None
    
    # Load the image
    img, data = load_mri_file(file_path)
    
    if img is not None:
        if show_info:
            print_image_info(img, data)
        
        fig = None
        if show_views:
            fig = display_3d_views(data, title=f"MRI Views: {os.path.basename(file_path)}")
        
        return img, data, fig
    
    return None, None, None 