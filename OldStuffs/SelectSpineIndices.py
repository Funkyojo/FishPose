import numpy as np

def select_spine_indices(Coords, m, canvas_sz):
    """
    Select the crystal shape around the spine coord (Coords)
    Coords format: np.array([(base)[i,j], (tip)[i,j]])
    m: margin around the spine_coord
    
    returns a Mask the same size of the original image, where the selected area has values of 1.
    """
    # Create a mesh of canvas, so we can index out the crystal shape
    I = np.array(range(0,canvas_sz[0]))
    J = np.array(range(0,canvas_sz[1]))
    ii, jj = np.meshgrid(I,J, indexing ='ij')

    t = Coords[np.argmin(Coords, axis=0)[0]]
    l = Coords[np.argmin(Coords, axis=0)[1]]
    b = Coords[np.argmax(Coords, axis=0)[0]]
    r = Coords[np.argmax(Coords, axis=0)[1]]

    # The four non-tip end points of the crystal shape
    l_aug = l - [0,m]
    r_aug = r + [0,m]
    t_aug = t - [m,0]
    b_aug = b + [m,0]

    # Determine if the spine has positive or negative slope
    if (l == t).all():
        Group_L = [t_aug, l_aug]
        Group_R = [b_aug, r_aug]
        Group_L_t = t_aug
        Group_L_b = l_aug
    else:
        Group_L = [b_aug, l_aug]
        Group_R = [t_aug, r_aug]
        Group_L_t = l_aug
        Group_L_b = b_aug

    # Slope of the crystal shape edge
    s = Group_R[0] - Group_L[1]
    s = s[0]/s[1]

    # Using conditions to map out the area.
    M = np.all([ii > t_aug[0], ii > Group_L_t[0] + (jj - Group_L_t[1])*s, 
                jj > l_aug[1], jj > l_aug[1] + (ii - l_aug[0])/s,
                ii < b_aug[0], ii < Group_L_b[0] + (jj - Group_L_b[1])*s,
                jj < r_aug[1], jj < r_aug[1] + (ii - r_aug[0])/s], axis=0)
    return M
